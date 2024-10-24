use pokervis::card::*;
use pokervis::holdem::*;



fn main() {
    let mut game = HoldemGame::new(vec![
        Player::new("ALC".to_string()),
        Player::new("BOB".to_string()),
        // Player::new("CHR".to_string()),
        //Player::new("DVD".to_string()),
        //Player::new("EVE".to_string()),
    ]);

    game.play_rounds(50, true);
    
    // test_equity();
    // print_preflop_range(|i| (PREFLOP_EQUITIES[i] * 9.5).round() as u8);
}
