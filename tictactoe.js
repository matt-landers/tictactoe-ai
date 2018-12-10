"use strict";
exports.__esModule = true;
var tf = require("@tensorflow/tfjs");
/*
Manages a game of TicTacToe with an array:
0 = empty slot
1 = X
-1 = O
The board is setup as a 9 slot array.
The numbers represent the index of each board position:
[
    0,1,2,
    3,4,5,
    6,7,8
]
Each board position will either be -1,0,1
*/
var TicTacToe = /** @class */ (function () {
    function TicTacToe() {
        var _this = this;
        this.PlayerTurn = Players.X;
        this.BoardState = function () { return tf.tensor2d([_this.boardState]); };
        this.GameState = GameStates.Playing;
        this.boardState = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        this.winningSlots = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [6, 7, 8],
            [0, 4, 5], [2, 4, 6] //Diagonal
        ];
    }
    TicTacToe.prototype.performMove = function (player, slot) {
        this.checkTurn(player);
        this.boardState[slot] = player;
        this.GameState = this.checkForWinnerOrDraw();
        this.PlayerTurn = player === Players.X ? Players.O : Players.X;
        return this.GameState;
    };
    TicTacToe.prototype.performRandomMove = function (player) {
        var openSlots = [];
        for (var i = 0; i < this.boardState.length; i++) {
            if (this.slotState(i) === 0) {
                openSlots.push(i);
            }
        }
        var slot = Math.floor((Math.random() * openSlots.length));
        this.performMove(player, openSlots[slot]);
    };
    TicTacToe.prototype.checkTurn = function (player) {
        if (this.PlayerTurn !== player) {
            throw "It is not this player's turn.";
        }
    };
    TicTacToe.prototype.checkForWinnerOrDraw = function () {
        for (var i = 0; i < this.winningSlots.length; i++) {
            var player = this.slotsHaveWinner(this.winningSlots[i]);
            if (player)
                return player;
        }
        if (!this.boardHasMoves())
            return GameStates.Draw;
        return GameStates.Playing;
    };
    TicTacToe.prototype.slotsHaveWinner = function (slots) {
        if (!this.slotsHavePlayer(slots))
            return false;
        var player = this.slotState(slots[0]);
        for (var i = 1; i < slots.length; i++) {
            var slot = slots[i];
            if (player !== this.slotState(slot))
                return false;
        }
        return player;
    };
    TicTacToe.prototype.slotsHavePlayer = function (slots) {
        for (var i = 0; i < slots.length; i++) {
            if (this.slotState(slots[i]) === 0)
                return false;
        }
        return true;
    };
    TicTacToe.prototype.boardHasMoves = function () {
        for (var i = 0; i < this.boardState.length; i++) {
            if (this.slotState(i) === 0)
                return true;
        }
        return false;
    };
    TicTacToe.prototype.slotState = function (slot) {
        return this.boardState[slot];
    };
    return TicTacToe;
}());
exports["default"] = TicTacToe;
var GameStates;
(function (GameStates) {
    GameStates[GameStates["Draw"] = 0] = "Draw";
    GameStates[GameStates["WinnerX"] = 1] = "WinnerX";
    GameStates[GameStates["WinnerO"] = -1] = "WinnerO";
    GameStates[GameStates["Playing"] = 2] = "Playing";
})(GameStates = exports.GameStates || (exports.GameStates = {}));
var Players;
(function (Players) {
    Players[Players["X"] = 1] = "X";
    Players[Players["O"] = -1] = "O";
})(Players = exports.Players || (exports.Players = {}));
