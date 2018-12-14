import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import embed from 'vega-embed';

import TicTacToe, { GameStates, Players } from './tictactoe';
import { SaveablePolicyNetwork } from './policygradient';
import { mean, sum } from './utils';

const appStatus = document.getElementById('app-status');
const storedModelStatusInput = <HTMLInputElement>document.getElementById('stored-model-status');
const hiddenLayerSizesInput = <HTMLInputElement>document.getElementById('hidden-layer-sizes');
const createModelButton = <HTMLButtonElement>document.getElementById('create-model');
const deleteStoredModelButton = <HTMLButtonElement>document.getElementById('delete-stored-model');

const numIterationsInput = <HTMLInputElement>document.getElementById('num-iterations');
const gamesPerIterationInput = <HTMLInputElement>document.getElementById('games-per-iteration');
const discountRateInput = <HTMLInputElement>document.getElementById('discount-rate');
const learningRateInput = <HTMLInputElement>document.getElementById('learning-rate');
const renderDuringTrainingCheckbox = <HTMLInputElement>document.getElementById('render-during-training');

const trainButton = <HTMLButtonElement>document.getElementById('train');
const testButton = <HTMLButtonElement>document.getElementById('test');
const iterationStatus = document.getElementById('iteration-status');
const iterationProgress = <HTMLProgressElement>document.getElementById('iteration-progress');
const trainStatus = document.getElementById('train-status');
const trainSpeed = document.getElementById('train-speed');
const trainProgress = <HTMLProgressElement>document.getElementById('train-progress');

const stepsContainer = document.getElementById('steps-container');

const ticTacToeBoard = document.getElementById('tictactoe-board');

let policyNet: SaveablePolicyNetwork;
let stopRequested = false;
let winsPerIteration: Array<any> = [];
const ticTacToe = new TicTacToe();

function logStatus(message) {
    appStatus!.textContent = message;
}

let renderDuringTraining = true;
export async function maybeRenderDuringTraining(ticTacToe) {
    if (renderDuringTraining) {
        //renderTicTacToe
        await tf.nextFrame();  // Unblock UI thread.
    }
}

// Objects and function to support the plotting of game steps during training.
function plotSteps() {
    tfvis.render.linechart({ values: winsPerIteration }, stepsContainer!, {
        xLabel: 'Training Iteration',
        yLabel: 'Wins Per Iteration',
        width: 400,
        height: 300,
    });
}

function onIterationEnd(iterationCount, totalIterations) {
    trainStatus!.textContent = `Iteration ${iterationCount} of ${totalIterations}`;
    trainProgress!.value = iterationCount / totalIterations * 100;
}

export function onGameEnd(gameCount, totalGames) {
    iterationStatus!.textContent = `Game ${gameCount} of ${totalGames}`;
    iterationProgress!.value = gameCount / totalGames * 100;
    if (gameCount === totalGames) {
        iterationStatus!.textContent = 'Updating weights...';
    }
}

async function updateUIControlState() {
    const modelInfo = await SaveablePolicyNetwork.checkStoredModelStatus();
    if (modelInfo == null) {
        storedModelStatusInput.value = 'No stored model.';
        deleteStoredModelButton!.disabled = true;

    } else {
        storedModelStatusInput.value = `Saved@${modelInfo.dateSaved.toISOString()}`;
        deleteStoredModelButton.disabled = false;
        createModelButton.disabled = true;
    }
    createModelButton.disabled = policyNet != null;
    hiddenLayerSizesInput.disabled = policyNet != null;
    trainButton.disabled = policyNet == null;
    testButton.disabled = policyNet == null;
    renderDuringTrainingCheckbox.checked = renderDuringTraining;
}

function disableModelControls() {
    trainButton.textContent = 'Stop';
    testButton.disabled = true;
    deleteStoredModelButton.disabled = true;
}

function enableModelControls() {
    trainButton.textContent = 'Train';
    testButton.disabled = false;
    deleteStoredModelButton.disabled = false;
}

function renderTicTacToe(ttt: TicTacToe) {
    
    const getPlayer = (state) => {
        switch (state) {
            case 1:
                return 'X';
            case -1:
                return 'O';
            default:
                return '&nbsp;';
        }
    }

    let values = ttt.BoardState().dataSync();
    let arr = Array.from(values);
    
    let table = `<table border="1">
        <tr><td>${getPlayer(arr[0])}</td><td>${getPlayer(arr[1])}</td><td>${getPlayer(arr[2])}</td></tr>
        <tr><td>${getPlayer(arr[3])}</td><td>${getPlayer(arr[4])}</td><td>${getPlayer(arr[5])}</td></tr>
        <tr><td>${getPlayer(arr[6])}</td><td>${getPlayer(arr[7])}</td><td>${getPlayer(arr[8])}</td></tr>
    </table>`;

    ticTacToeBoard.innerHTML = table;
}

export async function setUpUI() {

    if (await SaveablePolicyNetwork.checkStoredModelStatus() != null) {
        policyNet = await SaveablePolicyNetwork.loadModel();
        logStatus('Loaded policy network from IndexedDB.');
        hiddenLayerSizesInput.value = policyNet.hiddenLayerSizes();
    }

    await updateUIControlState();

    renderDuringTrainingCheckbox.addEventListener('change', () => {
        renderDuringTraining = renderDuringTrainingCheckbox.checked;
    });

    createModelButton!.addEventListener('click', async () => {
        try {
            const hiddenLayerSizes =
                hiddenLayerSizesInput!.value.trim().split(',').map(v => {
                    const num = parseInt(v.trim());
                    if (!(num > 0)) {
                        throw new Error(
                            `Invalid hidden layer sizes string: ` +
                            `${hiddenLayerSizesInput.value}`);
                    }
                    return num;
                });
            policyNet = new SaveablePolicyNetwork(hiddenLayerSizes);
            console.log('DONE constructing new instance of SaveablePolicyNetwork');
            await updateUIControlState();
        } catch (err) {
            logStatus(`ERROR: ${err.message}`);
        }
    });

    deleteStoredModelButton.addEventListener('click', async () => {
        if (confirm(`Are you sure you want to delete the locally-stored model?`)) {
            await policyNet.removeModel();
            policyNet = null;
            await updateUIControlState();
        }
    });

    trainButton.addEventListener('click', async () => {
        if (trainButton.textContent === 'Stop') {
            stopRequested = true;
        } else {
            disableModelControls();

            try {
                const trainIterations = Number.parseInt(numIterationsInput.value);
                if (!(trainIterations > 0)) {
                    throw new Error(`Invalid number of iterations: ${trainIterations}`);
                }
                const gamesPerIteration = Number.parseInt(gamesPerIterationInput.value);
                if (!(gamesPerIteration > 0)) {
                    throw new Error(
                        `Invalid # of games per iterations: ${gamesPerIteration}`);
                }
                const discountRate = Number.parseFloat(discountRateInput.value);
                if (!(discountRate > 0 && discountRate < 1)) {
                    throw new Error(`Invalid discount rate: ${discountRate}`);
                }
                const learningRate = Number.parseFloat(learningRateInput.value);

                logStatus(
                    'Training policy network... Please wait. ' +
                    'Network is saved to IndexedDB at the end of each iteration.');

                const optimizer = tf.train.adam(learningRate);

                winsPerIteration = [];
                onIterationEnd(0, trainIterations);
                let t0 = new Date().getTime();
                stopRequested = false;
                for (let i = 0; i < trainIterations; ++i) {
                    const gameWins = await policyNet.train(
                        ticTacToe, optimizer, discountRate, gamesPerIteration);
                    const t1 = new Date().getTime();
        
                    t0 = t1;
                    trainSpeed!.textContent = `Wins this iteration ${gameWins}`
                    winsPerIteration.push({ x: i + 1, y: gameWins });
                    console.log(`# of tensors: ${tf.memory().numTensors}`);
                    plotSteps();
                    onIterationEnd(i + 1, trainIterations);
                    await tf.nextFrame();  // Unblock UI thread.
                    await policyNet.saveModel();
                    await updateUIControlState();

                    if (stopRequested) {
                        logStatus('Training stopped by user.');
                        break;
                    }
                }
                if (!stopRequested) {
                    logStatus('Training completed.');
                }
            } catch (err) {
                logStatus(`ERROR: ${err.message}`);
            }
            enableModelControls();
        }
    });

    testButton.addEventListener('click', async () => {
        disableModelControls();
        let isDone = false;
        const ticTacToe = new TicTacToe();
        let steps = 0;
        stopRequested = false;
        while (ticTacToe.GameState === GameStates.Playing) {
            steps++;
            tf.tidy(() => {
                const action = policyNet.getActions(ticTacToe.BoardState())[0];
                
                logStatus(`Test in progress.`);
                ticTacToe.performMove(Players.X,action);
                if(ticTacToe.GameState === GameStates.Playing) ticTacToe.performRandomMove(Players.O);
                renderTicTacToe(ticTacToe);
            });
            await tf.nextFrame();  // Unblock UI thread.
            if (stopRequested) {
                break;
            }
        }
        if (stopRequested) {
            logStatus(`Test stopped by user after ${steps} step(s).`);
        } else {
            logStatus(`Test finished. Survived ${steps} step(s).`);
        }
        console.log(`# of tensors: ${tf.memory().numTensors}`);
        enableModelControls();
    });

}