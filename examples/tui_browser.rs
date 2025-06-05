use std::{collections::HashSet, env, fs, io, process::Command, thread, time::Duration};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseEventKind},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
    tty::IsTty,
};

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState},
    Terminal,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    if !io::stdout().is_tty() {
        eprintln!("No terminal detected. Exiting.");
        std::process::exit(1);
    }
    // Process command-line arguments.
    let args: Vec<String> = env::args().collect();
    let mut mode = "cargo_run"; // default mode
    let mut src_dir = "src".to_string();
    for arg in args.iter().skip(1) {
        if arg == "--src" {
            mode = "src";
            src_dir = "src".to_string();
        }
    }

    // Get list of available binaries.
    let mut pieces: Vec<String> = Vec::new();
    if mode == "src" {
        pieces = fs::read_dir(&src_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "rs" {
                            let stem = path.file_stem()?.to_string_lossy().to_string();
                            if stem != "main" && stem != "lib" {
                                return Some(stem);
                            }
                        }
                    }
                }
                None
            })
            .collect();
    } else {
        let output = Command::new("cargo").arg("run").output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        if let Some(index) = stderr.find("available binaries:") {
            let binaries_str = &stderr[index..];
            if let Some(colon_index) = binaries_str.find(':') {
                let bin_list = &binaries_str[colon_index + 1..];
                pieces = bin_list
                    .split(|c| c == ',' || c == '\n')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }
    pieces.sort();
    if pieces.is_empty() {
        println!("No shaders found using mode {}!", mode);
        return Ok(());
    }

    // Determine the directory where Cargo.toml resides.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let history_path = format!("{}/run_history.txt", manifest_dir);

    // Load run history from file into a HashSet.
    let mut run_history: HashSet<String> = HashSet::new();
    if let Ok(contents) = fs::read_to_string(&history_path) {
        for line in contents.lines() {
            if !line.trim().is_empty() {
                run_history.insert(line.trim().to_string());
            }
        }
    }

    // Set up terminal in raw mode with an alternate screen.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        Clear(ClearType::All)
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // ListState to track the selected shader.
    let mut list_state = ListState::default();
    list_state.select(Some(0));

    // Track whether the mouse is hovering over the right-side text.
    let mut exit_hover = false;

    'main_loop: loop {
        terminal.draw(|f| {
            // Get the full available drawing area.
            let size = f.area();
            let area = Rect::new(0, 0, size.width, size.height);
            // Create a single chunk (with margin) that will contain our list with a block title.
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints([Constraint::Min(0)].as_ref())
                .split(area);
            let list_area = chunks[0];

            // Compose a single-line title.
            let left_text = format!("Select shader ({} shaders found)", pieces.len());
            let separator = " ┃ ";
            // Right text as a whole.
            let right_text = "Esc or q to EXIT";
            let right_span = if exit_hover {
                // When hovered, the entire right text becomes yellow.
                Span::styled(right_text, Style::default().fg(Color::Yellow))
            } else {
                // Otherwise, "Esc or q to " is white and "EXIT" is red.
                // We'll combine them into two spans.
                // (They will appear adjacent.)
                // Note: The length of the entire string is used for mouse hit detection.
                // You can adjust the styles as needed.
                // Here we leave them as separate spans.
                Span::raw("") // placeholder; we'll build a vector below.
            };

            let title_line = if exit_hover {
                // Single span for the right text.
                Line::from(vec![
                    Span::raw(left_text),
                    Span::raw(separator),
                    right_span,
                ])
            } else {
                // Two spans for the right text.
                Line::from(vec![
                    Span::raw(left_text),
                    Span::raw(separator),
                    Span::styled("Esc or q to ", Style::default().fg(Color::White)),
                    Span::styled("EXIT", Style::default().fg(Color::Red)),
                ])
            };

            // Build the block with borders and the composite title.
            let block = Block::default().borders(Borders::ALL).title(title_line);

            // Build the list of shaders.
            let items: Vec<ListItem> = pieces
                .iter()
                .map(|p| {
                    let mut item = ListItem::new(p.as_str());
                    if run_history.contains(p) {
                        item = item.style(Style::default().fg(Color::Blue));
                    }
                    item
                })
                .collect();
            let list = List::new(items)
                .block(block)
                .highlight_style(Style::default().fg(Color::Yellow))
                .highlight_symbol(">> ");
            f.render_stateful_widget(list, list_area, &mut list_state);
        })?;

        // Poll for events.
        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break 'main_loop,
                    KeyCode::Down => {
                        let i = match list_state.selected() {
                            Some(i) if i >= pieces.len() - 1 => i,
                            Some(i) => i + 1,
                            None => 0,
                        };
                        list_state.select(Some(i));
                    }
                    KeyCode::Up => {
                        let i = match list_state.selected() {
                            Some(0) | None => 0,
                            Some(i) => i - 1,
                        };
                        list_state.select(Some(i));
                    }
                    KeyCode::Enter => {
                        if let Some(selected) = list_state.selected() {
                            run_piece(
                                &pieces,
                                selected,
                                &history_path,
                                &mut run_history,
                                &mut terminal,
                            )?;
                        }
                    }
                    _ => {}
                },
                Event::Mouse(mouse_event) => {
                    // Recompute layout to determine the block title area.
                    let size = terminal.size()?;
                    let area = Rect::new(0, 0, size.width, size.height);
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .margin(2)
                        .constraints([Constraint::Min(0)].as_ref())
                        .split(area);
                    let list_area = chunks[0];
                    // We assume the block title is drawn on the top border row of the block.
                    let title_row = list_area.y; // top border row
                    let title_start = list_area.x + 2; // assumed starting x-position of title text

                    // Compute the offset (in characters) for the right text.
                    let left_text = format!("Select shader ({} shaders found)", pieces.len());
                    let separator = " ┃ ";
                    let right_text = "Esc or q to EXIT";
                    let offset = (left_text.len() + separator.len()) as u16;
                    let right_region_start = title_start + offset;
                    let right_region_end = right_region_start + (right_text.len() as u16);

                    match mouse_event.kind {
                        MouseEventKind::Moved => {
                            if mouse_event.row == title_row {
                                if mouse_event.column >= right_region_start
                                    && mouse_event.column < right_region_end
                                {
                                    exit_hover = true;
                                } else {
                                    exit_hover = false;
                                }
                            } else {
                                exit_hover = false;
                                // Also update list selection if hovering over list area.
                                let inner_y = list_area.y + 1;
                                let inner_height = list_area.height.saturating_sub(2);
                                if mouse_event.column >= list_area.x + 1
                                    && mouse_event.column < list_area.x + list_area.width - 1
                                    && mouse_event.row >= inner_y
                                    && mouse_event.row < inner_y + inner_height
                                {
                                    let index = (mouse_event.row - inner_y) as usize;
                                    if index < pieces.len() {
                                        list_state.select(Some(index));
                                    }
                                }
                            }
                        }
                        MouseEventKind::Down(_) => {
                            if mouse_event.row == title_row
                                && mouse_event.column >= right_region_start
                                && mouse_event.column < right_region_end
                            {
                                break 'main_loop;
                            }
                            // Otherwise, if clicking in the list area, update selection and run.
                            let inner_y = list_area.y + 1;
                            let inner_height = list_area.height.saturating_sub(2);
                            if mouse_event.column >= list_area.x + 1
                                && mouse_event.column < list_area.x + list_area.width - 1
                                && mouse_event.row >= inner_y
                                && mouse_event.row < inner_y + inner_height
                            {
                                let index = (mouse_event.row - inner_y) as usize;
                                if index < pieces.len() {
                                    list_state.select(Some(index));
                                    run_piece(
                                        &pieces,
                                        index,
                                        &history_path,
                                        &mut run_history,
                                        &mut terminal,
                                    )?;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    // Restore terminal on exit.
    disable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        LeaveAlternateScreen,
        DisableMouseCapture,
        Clear(ClearType::All)
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// Runs the selected shader by executing the external command.
fn run_piece(
    pieces: &Vec<String>,
    index: usize,
    history_path: &str,
    run_history: &mut HashSet<String>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let piece = &pieces[index];
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    println!("Running: cargo run --release --bin {}", piece);
    let status = Command::new("cargo")
        .args(&["run", "--release", "--bin", piece])
        .status()?;
    println!("Process exited with status: {}\n", status);

    if run_history.insert(piece.clone()) {
        let history_data = run_history.iter().cloned().collect::<Vec<_>>().join("\n");
        fs::write(history_path, history_data)?;
    }

    while event::poll(Duration::from_millis(0))? {
        let _ = event::read();
    }
    thread::sleep(Duration::from_millis(50));
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        Clear(ClearType::All)
    )?;
    *terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    Ok(())
}

