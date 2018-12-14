import * as tf from '@tensorflow/tfjs';

import { maybeRenderDuringTraining, onGameEnd } from './ui';
import TicTacToe, { Players, GameStates } from './tictactoe';

/**
 * Policy network for controlling the tic-tac-toe game.
  */
class PolicyNetwork {
    protected model: any;
    private currentActions_: any;
    /**
    * Constructor of PolicyNetwork.
    *
    * @param {number | number[] | tf.Model} hiddenLayerSizes
    *   Can be any of the following
    *   - Size of the hidden layer, as a single number (for a single hidden
    *     layer)
    *   - An Array of numbers (for any number of hidden layers).
    *   - An instance of tf.Model.
    */

    constructor(hiddenLayerSizesOrModel) {
        if (hiddenLayerSizesOrModel instanceof tf.Model) {
            this.model = hiddenLayerSizesOrModel;
        } else {
            this.createModel(hiddenLayerSizesOrModel);
        }
    }

    /**
     * Create the underlying model of this policy network.
     *
     * @param {number | number[]} hiddenLayerSizes Size of the hidden layer, as
     *   a single number (for a single hidden layer) or an Array of numbers (for
     *   any number of hidden layers).
     */
    createModel(hiddenLayerSizes) {
        if (!Array.isArray(hiddenLayerSizes)) {
            hiddenLayerSizes = [hiddenLayerSizes];
        }
        this.model = tf.sequential();
        hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
            this.model!.add(tf.layers.dense({
                units: hiddenLayerSize,
                activation: 'elu',
                // `inputShape` is required only for the first layer.
                inputShape: i === 0 ? [9] : undefined
            }));
        });
        // The last layer has only one unit. The single output number will be
        // converted to a probability of selecting the leftward-force action.
        this.model.add(tf.layers.dense({ units: 9, activation: 'softmax' }));
    }

    /**
     * Train the policy network's model.
     *
     * @param {CartPole} cartPoleSystem The cart-pole system object to use during
     *   training.
     * @param {tf.train.Optimizer} optimizer An instance of TensorFlow.js
     *   Optimizer to use for training.
     * @param {number} discountRate Reward discounting rate: a number between 0
     *   and 1.
     * @param {number} numGames Number of game to play for each model parameter
     *   update.
     * @returns {number[]} The number of steps completed in the `numGames` games
     *   in this round of training.
     */
    async train(
        ticTacToe: TicTacToe, optimizer, discountRate, numGames) {
        const allGradients = [];
        const allRewards = [];
        let gameWins = 0;
        onGameEnd(0, numGames);
        for (let i = 0; i < numGames; ++i) {
            // Randomly initialize the state of the cart-pole system at the beginning
            // of every game.
            ticTacToe = new TicTacToe();
            const gameRewards = [];
            const gameGradients = [];
            for (let j = 0; j <= 9; ++j) { //loop 9 times because that is all the moves there are in tic tac toe
                // For every step of the game, remember gradients of the policy
                // network's weights with respect to the probability of the action
                // choice that lead to the reward.
                
                const gradients = tf.tidy(() => {
                    const inputTensor = ticTacToe.BoardState();
                    return this.getGradientsAndSaveActions(inputTensor).grads;
                });
                
                this.pushGradients(gameGradients, gradients);
                const action = this.currentActions_[0];
                //console.log(this.currentActions_);
                ticTacToe.performMove(Players.X, action);
                
                //Play a random move for O
                if(ticTacToe.GameState === GameStates.Playing) ticTacToe.performRandomMove(Players.O);

                await maybeRenderDuringTraining(ticTacToe);
                
                if (ticTacToe.GameState !== GameStates.Playing) {
                    // When the game ends before max step count is reached, a reward of
                    // 0 is given.
                    if(ticTacToe.GameState === GameStates.WinnerX) {
                        gameRewards.push(1);
                        console.log(`Wins: ${gameWins++}`);
                    } else {
                        gameRewards.push(0);
                    }
                    
                    break;
                }
                
            }
            onGameEnd(i + 1, numGames);
            this.pushGradients(allGradients, gameGradients);
            allRewards.push(gameRewards);
            await tf.nextFrame();
        }

        tf.tidy(() => {
            // The following line does three things:
            // 1. Performs reward discounting, i.e., make recent rewards count more
            //    than rewards from the further past. The effect is that the reward
            //    values from a game with many steps become larger than the values
            //    from a game with fewer steps.
            // 2. Normalize the rewards, i.e., subtract the global mean value of the
            //    rewards and divide the result by the global standard deviation of
            //    the rewards. Together with step 1, this makes the rewards from
            //    long-lasting games positive and rewards from short-lasting
            //    negative.
            // 3. Scale the gradients with the normalized reward values.
            const normalizedRewards =
                discountAndNormalizeRewards(allRewards, discountRate);
            // Add the scaled gradients to the weights of the policy network. This
            // step makes the policy network more likely to make choices that lead
            // to long-lasting games in the future (i.e., the crux of this RL
            // algorithm.)
            optimizer.applyGradients(
                scaleAndAverageGradients(allGradients, normalizedRewards));
        });
        tf.dispose(allGradients);
        return gameWins;
    }

    getGradientsAndSaveActions(inputTensor) {
        const f = () => tf.tidy(() => {
            const [logits, actions] = this.getLogitsAndActions(inputTensor);
            this.currentActions_ = actions.dataSync();
            const labels =
                tf.sub(1, tf.tensor(this.currentActions_, actions.shape));
            return tf.losses.sigmoidCrossEntropy(labels, logits).asScalar();
        });
        return tf.variableGrads(f);
    }

    getCurrentActions() {
        return this.currentActions_;
    }

    /**
     * Get policy-network logits and the action based on state-tensor inputs.
     *
     * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 9]`.
     * @returns {[tf.Tensor, tf.Tensor]}
     *   1. The logits tensor, of shape `[batchSize, 1]`.
     *   2. The actions tensor, of shape `[batchSize, 1]`.
     */
    getLogitsAndActions(inputs) {
        return tf.tidy(() => {
            //console.log(`inputs: ${inputs}`);
            const logits = this.model.predict(inputs);
            //console.log(`logits: ${logits}`);
            // Get the probability of the leftward action.
            //const sigmoid = tf.sigmoid(logits);
            
            //console.log(`sigmoid: ${sigmoid}`)
            // Probabilites of the left and right actions.
            //const leftRightProbs = tf.concat([sigmoid, tf.sub(1, sigmoid)], 1);
            //console.log(`concat: ${leftRightProbs}`);
            let values = logits.dataSync();
            //console.log(values);
            const probs = tf.multinomial(values, 9, undefined, true);
            

            const actions = probs.reshape([1,9])
            //console.log(`actions: ${actions}`);
            return [logits, actions];
        });
    }

    /**
     * Get actions based on a state-tensor input.
     *
     * @param {tf.Tensor} inputs A tf.Tensor instance of shape `[batchSize, 4]`.
     * @param {Float32Array} inputs The actions for the inputs, with length
     *   `batchSize`.
     */
    getActions(inputs) {
        return this.getLogitsAndActions(inputs)[1].dataSync();
    }

    /**
     * Push a new dictionary of gradients into records.
     *
     * @param {{[varName: string]: tf.Tensor[]}} record The record of variable
     *   gradient: a map from variable name to the Array of gradient values for
     *   the variable.
     * @param {{[varName: string]: tf.Tensor}} gradients The new gradients to push
     *   into `record`: a map from variable name to the gradient Tensor.
     */
    pushGradients(record, gradients) {
        for (const key in gradients) {
            if (key in record) {
                record[key].push(gradients[key]);
            } else {
                record[key] = [gradients[key]];
            }
        }
    }
}

// The IndexedDB path where the model of the policy network will be saved.
const MODEL_SAVE_PATH_ = 'indexeddb://cart-pole-v1';

/**
 * A subclass of PolicyNetwork that supports saving and loading.
 */
export class SaveablePolicyNetwork extends PolicyNetwork {
    /**
     * Constructor of SaveablePolicyNetwork
     *
     * @param {number | number[]} hiddenLayerSizesOrModel
     */
    constructor(hiddenLayerSizesOrModel) {
        super(hiddenLayerSizesOrModel);
    }

    /**
     * Save the model to IndexedDB.
     */
    async saveModel() {
        return await this.model.save(MODEL_SAVE_PATH_);
    }

    /**
     * Load the model fom IndexedDB.
     *
     * @returns {SaveablePolicyNetwork} The instance of loaded
     *   `SaveablePolicyNetwork`.
     * @throws {Error} If no model can be found in IndexedDB.
     */
    static async loadModel() {
        const modelsInfo = await tf.io.listModels();
        if (MODEL_SAVE_PATH_ in modelsInfo) {
            console.log(`Loading existing model...`);
            const model = await tf.loadModel(MODEL_SAVE_PATH_);
            console.log(`Loaded model from ${MODEL_SAVE_PATH_}`);
            return new SaveablePolicyNetwork(model);
        } else {
            throw new Error(`Cannot find model at ${MODEL_SAVE_PATH_}.`);
        }
    }

    /**
     * Check the status of locally saved model.
     *
     * @returns If the locally saved model exists, the model info as a JSON
     *   object. Else, `undefined`.
     */
    static async checkStoredModelStatus() {
        const modelsInfo = await tf.io.listModels();
        return modelsInfo[MODEL_SAVE_PATH_];
    }

    /**
     * Remove the locally saved model from IndexedDB.
     */
    async removeModel() {
        return await tf.io.removeModel(MODEL_SAVE_PATH_);
    }

    /**
     * Get the sizes of the hidden layers.
     *
     * @returns {number | number[]} If the model has only one hidden layer,
     *   return the size of the layer as a single number. If the model has
     *   multiple hidden layers, return the sizes as an Array of numbers.
     */
    hiddenLayerSizes() {
        const sizes = [];
        for (let i = 0; i < this.model.layers.length - 1; ++i) {
            sizes.push(this.model.layers[i].units);
        }
        return sizes.length === 1 ? sizes[0] : sizes;
    }
}

/**
 * Discount the reward values.
 *
 * @param {number[]} rewards The reward values to be discounted.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor} The discounted reward values as a 1D tf.Tensor.
 */
function discountRewards(rewards, discountRate) {
    const discountedBuffer = tf.buffer([rewards.length]);
    let prev = 0;
    for (let i = rewards.length - 1; i >= 0; --i) {
        const current = discountRate * prev + rewards[i];
        discountedBuffer.set(current, i);
        prev = current;
    }
    return discountedBuffer.toTensor();
}

/**
 * Discount and normalize reward values.
 *
 * This function performs two steps:
 *
 * 1. Discounts the reward values using `discountRate`.
 * 2. Normalize the reward values with the global reward mean and standard
 *    deviation.
 *
 * @param {number[][]} rewardSequences Sequences of reward values.
 * @param {number} discountRate Discount rate: a number between 0 and 1, e.g.,
 *   0.95.
 * @returns {tf.Tensor[]} The discounted and normalize reward values as an
 *   Array of tf.Tensor.
 */
function discountAndNormalizeRewards(rewardSequences, discountRate) {
    return tf.tidy(() => {
        const discounted = [];
        for (const sequence of rewardSequences) {
            discounted.push(discountRewards(sequence, discountRate))
        }
        // Compute the overall mean and stddev.
        const concatenated = tf.concat(discounted);
        const mean = tf.mean(concatenated);
        const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))));
        // Normalize the reward sequences using the mean and std.
        const normalized = discounted.map(rs => rs.sub(mean).div(std));
        return normalized;
    });
}

/**
 * Scale the gradient values using normalized reward values and compute average.
 *
 * The gradient values are scaled by the normalized reward values. Then they
 * are averaged across all games and all steps.
 *
 * @param {{[varName: string]: tf.Tensor[][]}} allGradients A map from variable
 *   name to all the gradient values for the variable across all games and all
 *   steps.
 * @param {tf.Tensor[]} normalizedRewards An Array of normalized reward values
 *   for all the games. Each element of the Array is a 1D tf.Tensor of which
 *   the length equals the number of steps in the game.
 * @returns {{[varName: string]: tf.Tensor}} Scaled and averaged gradients
 *   for the variables.
 */
function scaleAndAverageGradients(allGradients, normalizedRewards) {
    return tf.tidy(() => {
        const gradients = {};
        for (const varName in allGradients) {
            gradients[varName] = tf.tidy(() => {
                // Stack gradients together.
                const varGradients = allGradients[varName].map(
                    varGameGradients => tf.stack(varGameGradients));
                // Expand dimensions of reward tensors to prepare for multiplication
                // with broadcasting.
                const expandedDims = [];
                for (let i = 0; i < varGradients[0].rank - 1; ++i) {
                    expandedDims.push(1);
                }
                const reshapedNormalizedRewards = normalizedRewards.map(
                    rs => rs.reshape(rs.shape.concat(expandedDims)));
                for (let g = 0; g < varGradients.length; ++g) {
                    // This mul() call uses broadcasting.
                    varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
                }
                // Concatenate the scaled gradients together, then average them across
                // all the steps of all the games.
                return tf.mean(tf.concat(varGradients, 0), 0);
            });
        }
        return gradients;
    });
}