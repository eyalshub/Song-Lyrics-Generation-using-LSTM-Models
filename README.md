# Song Lyrics Generation using LSTM Models

This project explores the use of LSTM (Long Short-Term Memory) networks for generating song lyrics. By training the model on a dataset containing both lyrics and melodies, the goal is to develop a system that generates lyrics based on given melodies. The melodies, stored in MIDI format, are used alongside textual data to build a model that can create coherent song lyrics with an appropriate melody.

## Data Preparation

Effective data preparation is crucial for optimizing both the lyrics and melodies for machine learning tasks. Here are the preprocessing steps for each component:

### Lyrics Processing
1. **Cleaning and Special Character Removal**: 
   - Remove special characters and non-essential punctuation to focus on meaningful content.
   
2. **Tokenization and Case Standardization**:
   - Split the lyrics into tokens (e.g., words) and convert all text to lowercase to maintain consistency and reduce vocabulary size.
   
3. **Contractions Expansion**:
   - Expand abbreviations such as "isn't" to "is not" for standardization.
   
4. **Word Embeddings**:
   - Use pre-trained embeddings (e.g., Word2Vec) to convert words into dense vector representations that capture semantic relationships.

### Melody Processing (MIDI Data)
1. **Feature Extraction**: 
   - Extract key melodic features such as pitch, tempo, rhythm, and note durations from MIDI files.
   
2. **MIDI File Parsing**: 
   - Utilize the `pretty_midi` library to read MIDI files and extract essential musical data.
   
3. **Normalization and Sequence Padding**:
   - Normalize melodic features and pad sequences to ensure consistent input lengths for model training.

### Train-Validation Split
The dataset is split into training (80%) and validation (20%) sets to ensure robust model evaluation.

## Model Approach

The project uses two approaches, both of which utilize LSTM networks to learn relationships between song lyrics and melodies.

### Approach 1: Integrating General MIDI Features
This approach uses structural and timing-related features of MIDI data:
- **Resolution**: Number of ticks per time frame.
- **Key Signature Changes**: Tracks shifts in the musical key.
- **Timing Features**: Includes `tick_to_time` (mapping ticks to time) and `tick_scales` (scaling ticks to maintain consistent temporal representation).

#### Data Preprocessing
- MIDI features are processed to ensure consistency across samples and combined with Word2Vec embeddings for the lyrics.

### Approach 2: Integrating Sonic MIDI Features
This approach incorporates sonic features of the melody:
- **Pitches**: Frequencies of the notes.
- **Velocities**: Intensities of the notes.
- **Durations**: Length of each note.
- **Instruments**: Types of instruments used.

#### Data Preprocessing
- Sonic features are scaled and padded, similar to Approach 1, and combined with Word2Vec embeddings for the lyrics.

## Model Architecture: LSTM with Multihead Attention

The model architecture integrates LSTM layers with a Multihead Attention mechanism to predict song lyrics based on the combination of lyrics and MIDI features.

### Architecture Components
1. **LSTM Layers**:
   - Two LSTM layers capture temporal dependencies in the input data.
   - The first LSTM layer has an input size of 400 features (Word2Vec + MIDI) and outputs 256 features.
   - The second LSTM layer has an input size of 64 features and outputs 256 features.

2. **Dropout Layer**:
   - A dropout rate of 0.2 is used to prevent overfitting by deactivating a fraction of the network connections during training.

3. **Multihead Attention**:
   - Four attention heads are used to capture different parts of the input sequence simultaneously, improving contextual understanding.

4. **Layer Normalization**:
   - A normalization layer ensures stable training and prevents gradient explosion.

5. **Fully Connected Layer**:
   - The final dense layer maps the output to predict the next word in the sequence.

6. **Loss Function**:
   - Mean Squared Error (MSELoss) is used to evaluate the difference between predicted and actual word embeddings.

7. **Optimizer**:
   - The Adam optimizer with a weight decay of \( 1 \times 10^{-5} \) is used for efficient training.
   - A learning rate scheduler (ReduceLROnPlateau) dynamically adjusts the learning rate based on validation performance.

8. **Gradient Clipping**:
   - Gradients are clipped to a maximum value of 1.0 to stabilize training.

The alignment between lyrics and melodies is handled by the `collate_fn` function, which processes batches of data into tensors for model input and output.

## Evaluation

The model's performance is evaluated using several metrics:

### 1. Textual Similarity
- **Cosine Similarity**
- **Word Mean Distance (WMD)**
- **Sentence-BERT (SBERT)**

### 2. Lexical Overlap
- Measures the percentage of words in common between the original and generated lyrics.

### 3. Repetition Ratio
- Calculated as the proportion of unique words relative to the total word count in the generated text.

### 4. Quality Score Calculation
A weighted formula is used to calculate the quality score of the generated lyrics:
- Cosine Similarity: 30%
- Lexical Overlap: 20%
- Word Moving Distance (WMD): 30%
- Semantic Similarity (SBERT): 20%
- Repetition Ratio: 10%

## Generated Lyrics Examples

Here are some examples of generated lyrics along with the corresponding melody, cosine similarity, vocabulary overlap, and Word Mover's Distance.

### Melody: "Aqua - Barbie Girl"
- **Cosine Similarity**: 0.9176
- **Vocabulary Overlap**: 0.0490
- **Word Mover's Distance**: 2894.2860
- **Generated Lyrics**: 
