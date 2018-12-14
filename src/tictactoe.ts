import * as tf from '@tensorflow/tfjs';

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

export default class TicTacToe {
    
    public PlayerTurn = Players.X;
    public BoardState = () => tf.tensor2d([this.boardState]);
    public GameState: GameStates = GameStates.Playing;
    
    private boardState = [0,0,0,0,0,0,0,0,0];
    private winningSlots = [
        [0,1,2], [3,4,5], [6,7,8], //Horizontal
        [0,3,6], [1,4,7], [6,7,8], //Vertical
        [0,4,5], [2,4,6] //Diagonal
    ]
    
    performMove(player: Players, slot: number): GameStates {
        this.checkTurn(player);
        if(this.slotState(slot) === 0) this.boardState[slot] = player;
        this.GameState = this.checkForWinnerOrDraw();
        this.PlayerTurn = player === Players.X ? Players.O : Players.X;
        return this.GameState;
    }

    performRandomMove(player: Players) {
       
        let openSlots: Array<number> = [];
        for (let i = 0; i < this.boardState.length; i++) {
            if(this.slotState(i) === 0) {
                openSlots.push(i);
            }
        }

        let slot: number = Math.floor((Math.random() * openSlots.length));

        this.performMove(player, openSlots[slot]);
    }

    private checkTurn(player: Players) {
        if(this.PlayerTurn !== player) {
            throw `It is not this player's turn.`;
        }
    }

    private checkForWinnerOrDraw(): GameStates {

        for (let i = 0; i < this.winningSlots.length; i++) {
            let player = this.slotsHaveWinner(this.winningSlots[i]);
            if(player) return <GameStates><unknown>player;
        }

        if(!this.boardHasMoves()) return GameStates.Draw;

        return GameStates.Playing;
    }

    private slotsHaveWinner(slots: Array<number>) {
        
        if(!this.slotsHavePlayer(slots)) return false;

        let player: Players = this.slotState(slots[0]);
        
        for (let i = 1; i < slots.length; i++) {
            const slot = slots[i];
            if(player !== this.slotState(slot)) return false;
        }
        return player;
    }

    private slotsHavePlayer(slots: Array<number>) {
        for (let i = 0; i < slots.length; i++) {
            if(this.slotState(slots[i]) === 0) return false;
        }
        return true;
    }

    private boardHasMoves() {
        for (let i = 0; i < this.boardState.length; i++) {
            if(this.slotState(i) === 0) return true;
        }
        return false;
    }

    private slotState(slot: number) {
        return this.boardState[slot];
    }
}

export enum GameStates {
    Draw = 0,
    WinnerX = 1,
    WinnerO = -1,
    Playing = 2,
}
export enum Players {
    X = 1,
    O = -1
}
