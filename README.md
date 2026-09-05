# Deep Learning HW4: Recurrent Neural Networks and Time Series

This experiment comprehensively investigates recurrent neural network architectures, including vanilla RNNs, LSTMs, and GRUs, for multivariate time series forecasting.
It analyzes the impact of critical hyperparameters like hidden size, sequence length, and model depth on training stability, vanishing gradients, and long-term memory retention.  
Furthermore, it explores multi-step recursive forecasting and the conceptual limitations of bidirectional recurrent models in real-time sequential predictions.

**Datasets:** Synthetic multivariate time series with delayed dependencies.  
**Models:** Vanilla RNN, LSTM, GRU, and Bidirectional LSTM.

## Experiments

### Experiment 1: Dataset Feature Correlation and Temporal Dependencies
**Description**  
Five input features and one target variable were analyzed using correlation matrices and time-domain plots to identify underlying patterns.
The analysis aimed to discover direct correlations and delayed temporal dependencies between the target variable and the provided synthetic inputs.

**Achievements and Results**  
The target variable showed the highest direct correlation with feature x5, while feature x3 exhibited a clear delayed dependency on features x1 and x2.
These temporal delays confirmed that recurrent architectures are strictly necessary to effectively model the twenty-step historical dependencies within the dataset.

### Experiment 2: RNN Hidden Size Sensitivity and Overfitting
**Description**  
A single-layer RNN was trained with hidden sizes ranging from 8 to 256 to evaluate the trade-off between model capacity and generalization.
The test mean squared error and parameter counts were compared to identify the optimal representation bottleneck for the synthetic dataset.

**Achievements and Results**  
A hidden size of 16 provided the best generalization, while capacities of 128 and 256 led to severe overfitting and massive error spikes.
Increasing the hidden state size exponentially increased the parameter count but barely impacted training time due to the sequential GPU bottleneck.

### Experiment 3: RNN Sequence Length and Delay Capture
**Description**  
The input sequence length was varied from 10 to 200 steps to determine how much historical context a simple RNN can effectively leverage.
The experiment assessed whether extending the temporal window helps the network capture the known twenty-step delayed dependencies.

**Achievements and Results**  
A sequence length of 100 achieved the lowest error by providing sufficient context to capture all delayed correlations without introducing excessive noise.
Increasing the length further to 200 severely degraded performance, demonstrating the RNN's inability to filter irrelevant long-term noise.

### Experiment 4: Depth Limitations in Simple RNNs
**Description**  
Vanilla RNNs with one to four stacked layers were trained while keeping the hidden size and sequence length strictly constant.
The training stability, convergence speed, and final test errors were monitored to evaluate the benefits of hierarchical temporal feature extraction.

**Achievements and Results**  
The single-layer RNN successfully converged and achieved the lowest error, whereas all deeper configurations completely failed to learn meaningful representations.
The deeper networks exhibited severe training instability and high final errors due to the rapid vanishing of gradients across multiple recurrent layers.

### Experiment 5: LSTM vs RNN Capacity and Performance
**Description**  
LSTM networks with varying hidden sizes were trained and directly compared against their vanilla RNN counterparts under identical hyperparameters.
The comparison focused on analyzing whether the gating mechanisms justify the four-fold increase in trainable parameters.

**Achievements and Results**  
The LSTM consistently outperformed the RNN across all hidden sizes, demonstrating a much higher resistance to overfitting at larger capacities.
Even with a minimal hidden size of 8, the LSTM achieved near-optimal accuracy, proving its superior efficiency in modeling temporal data.

### Experiment 6: Long-Term Dependency Learning in LSTM
**Description**  
LSTM and RNN models were tested on sequence lengths varying from 10 to 100 to evaluate their handling of long-term temporal dependencies.
The experiment specifically measured how well each architecture utilizes restricted input windows versus extended historical contexts.

**Achievements and Results**  
The LSTM efficiently captured dependencies even with short sequences, reaching its peak performance at a sequence length of 50.
While the RNN required a window of 100 to minimize error, the LSTM overfit at this length due to the reduced number of valid training samples.

### Experiment 7: LSTM Cell State vs Hidden State Dynamics
**Description**  
The internal cell state and hidden state of a trained LSTM were extracted and plotted over time for a specific test sequence.
The visualization aimed to interpret the distinct roles of the additive memory track and the filtered output state.

**Achievements and Results**  
The cell state evolved smoothly and maintained a stable trajectory, confirming its primary role as the network's robust long-term memory.
Conversely, the hidden state exhibited sharp, erratic fluctuations, reacting immediately to local input changes to serve as a short-term output.

### Experiment 8: Performance Comparison of RNN, LSTM, and GRU
**Description**  
Vanilla RNN, LSTM, and GRU architectures were evaluated using an identical hidden size of 64 and a sequence length of 20.
Test errors, parameter counts, and overall training times were compared to identify the most balanced recurrent architecture.

**Achievements and Results**  
The LSTM achieved the lowest overall error but required the highest parameter count and the longest training time to converge.
Surprisingly, the GRU underperformed the simpler RNN in this specific configuration, likely due to suboptimal hyperparameters or dataset characteristics.

### Experiment 9: GRU Gate Activation Analysis
**Description**  
The average values of the update and reset gates in a trained GRU were extracted and visualized during inference on test data.
The activation patterns were analyzed to deduce how the network balances retaining historical context versus integrating new information.

**Achievements and Results**  
The update gate maintained a high average value above 0.7, indicating a strong preference for preserving long-term historical information.
The reset gate displayed deeper, more frequent dips, demonstrating the network's adaptive capability to clear short-term memory when encountering sudden shifts.

### Experiment 10: Deep RNN Vanishing Gradient Tracking
**Description**  
A five-layer vanilla RNN was trained, and the L2 norm of the gradients for the first and last recurrent layers were tracked per epoch.
The magnitude ratio between the layers was calculated to mathematically verify the presence of the vanishing gradient phenomenon.

**Achievements and Results**  
The gradient norm in the first layer rapidly approached zero, becoming thousands of times smaller than the gradient in the final layer.
This massive discrepancy empirically proved that early layers in deep vanilla RNNs suffer from vanishing gradients and fail to update effectively.

### Experiment 11: LSTM Superiority on Extremely Long Sequences
**Description**  
The sequence length was drastically increased up to 500 steps to test the extreme long-term memory limits of both RNN and LSTM models.
The final test errors were compared to see which architecture could withstand massive sequences without losing crucial historical signals.

**Achievements and Results**  
The LSTM maintained a perfectly stable and low error rate, utilizing its additive cell state to seamlessly propagate gradients across 500 steps.
In stark contrast, the vanilla RNN failed to improve and maintained a high error rate, proving its complete inability to handle extremely long sequences.

### Experiment 12: Gradient Clipping for Training Stability
**Description**  
Gradient clipping was applied to a deep RNN using various threshold values ranging from 0.5 to 5.0 to mitigate explosive gradient updates.
The training loss and gradient norms were monitored to identify the optimal threshold that prevents divergence without halting learning.

**Achievements and Results**  
A clipping threshold of 1.0 successfully stabilized the training process, preventing numerical explosions while allowing steady error reduction.
Smaller thresholds completely froze the learning process, while larger thresholds failed to tame the occasional gradient spikes, leading to model divergence.

### Experiment 13: Multi-Step Recursive Forecasting Resilience
**Description**  
The trained models were used to recursively predict future values at horizons of 1, 5, 10, 20, and 50 steps ahead.
The compound error accumulation was measured to evaluate which architecture best suppresses the exponential growth of recursive hallucinations.

**Achievements and Results**  
Error rates compounded significantly for all models as the forecasting horizon extended, completely degrading the predictions by step 50.
However, the GRU demonstrated the highest resilience to error accumulation, outperforming the LSTM which suffered heavily from overconfident long-term extrapolations.


## Extra/Optional Experiments

### Experiment 14: Conceptual Flaws of Bidirectional LSTM in Forecasting
**Description**  
A Bidirectional LSTM was trained and compared against a standard unidirectional LSTM with an equivalent total parameter count.
The experiment evaluated whether allowing the network to access future data during training improves sequential next-step forecasting.

**Achievements and Results**  
The Bidirectional LSTM yielded a higher error rate and required significantly more training time compared to the unidirectional baseline.
More importantly, the bidirectional approach was deemed conceptually invalid for time-series forecasting, as real-world inference strictly lacks access to future data.
