import TicTacToe, { GameStates, Players } from './tictactoe';

let tictactoe = new TicTacToe();

let currentPlayer = Players.X;

while(tictactoe.GameState === GameStates.Playing) {
    tictactoe.performRandomMove(tictactoe.PlayerTurn);    
}

console.log(tictactoe.GameState);