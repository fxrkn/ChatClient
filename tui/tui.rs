use std::io::stdout;

use crossterm::{
	event::{read, Event, KeyCode},
	terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
	cursor::MoveTo,
	ExecutableCommand,
};

use ratatui::{backend::CrosstermBackend, widgets::Paragraph, widgets::{Clear as ClearWidget, Block, Borders}, Terminal,};

use ratatui::layout::{Layout, Constraint, Direction};

struct App{
	sayac: u32,
	son_tus: String,
}

fn main() -> std::io::Result<()>{

	let mut out = stdout();
	out.execute(Clear(ClearType::All))?.execute(MoveTo(0, 0))?;

	enable_raw_mode()?;

	let mut app = App{
		sayac: 0,
		son_tus: String::from("nope"),
	};

	loop{

		let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

		terminal.draw(|frame|{

		let ana_parcalar = Layout::default()
			.direction(Direction::Vertical)
			.constraints([
				Constraint::Min(1),
				Constraint::Length(1),
				])
			.split(frame.size());

			let cerceve = Block::default().borders(Borders::ALL);


			let metin = format!(
				"Sayac: {}\nSon tus: {}\nCikmak icin 'q'",
				app.sayac,
				app.son_tus
				);

			let paragraph = Paragraph::new(metin).block(cerceve);
			let vga_area = ratatui::layout::Rect::new(0, 0, 80.min(frame.size().width), 25.min(frame.size().height));

			frame.render_widget(ClearWidget, vga_area);
			frame.render_widget(paragraph, ana_parcalar[0]);
			let alt_bilgi = Paragraph::new(" [F1] Cik");
			frame.render_widget(alt_bilgi, ana_parcalar[1]);
		})?;

		if let Event::Key(tus_olayi) = read()?{
			if tus_olayi.code == KeyCode::F(1){
				break;
			}
			app.sayac += 1;
			app.son_tus = format!("{:?}", tus_olayi.code);
		}
	}

	disable_raw_mode()?;

	let mut out = stdout();
	out.execute(Clear(ClearType::All))?.execute(MoveTo(0, 0))?;


	Ok(())
}
