"use strict";
exports.__esModule = true;
var tictactoe_1 = require("./tictactoe");
var tictactoe = new tictactoe_1["default"]();
var currentPlayer = tictactoe_1.Players.X;
while (tictactoe.GameState === tictactoe_1.GameStates.Playing) {
    tictactoe.performRandomMove(tictactoe.PlayerTurn);
}
console.log(tictactoe.GameState);
