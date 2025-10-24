# Fine-Tuning TinyLlama for News Article Summarization 📰

## 🚀 Overview

This project provides a comprehensive walkthrough for fine-tuning the **TinyLlama-1.1B-Chat-v1.0** language model specifically for the task of **abstractive text summarization**. It utilizes the well-known **CNN/DailyMail dataset** as the training corpus.

The primary goal is to demonstrate how a relatively small language model can be effectively adapted for a complex NLP task using modern, efficient techniques, making advanced model customization accessible even in resource-constrained environments (like Google Colab or Kaggle notebooks with a single GPU).

## ✨ Features

* **Efficient Fine-Tuning:** Employs **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** via the Hugging Face `peft` library. This allows significant adaptation of the model by training only a small fraction of its parameters.
* **Memory Optimization:** Integrates **4-bit quantization** using the `bitsandbytes` library. This drastically reduces the model's memory footprint, enabling fine-tuning on GPUs with limited VRAM (e.g., <= 16GB).
* **Automated Hyperparameter Search:** Leverages **Ray Tune** to systematically explore and identify optimal hyperparameters (like learning rate, gradient accumulation steps, weight decay) based on validation loss and ROUGE scores, removing guesswork.
* **Comprehensive Evaluation:** Measures performance using both standard **cross-entropy loss** and **perplexity**, alongside task-specific **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L) to assess the quality of generated summaries.
* **Benchmarking:** Compares the optimized fine-tuned model against a baseline (using default hyperparameters) and established **State-of-the-Art (SOTA)** models (BART-Large, PEGASUS) on the CNN/DailyMail dataset.
* **Qualitative Analysis:** Includes an error analysis section to identify common patterns in the generated summaries (e.g., length mismatches, coherence issues).
* **Sample Prediction:** Demonstrates the final model's summarization capability on a new, unseen article.

## 📊 Dataset: CNN/DailyMail

The **CNN/DailyMail dataset** is a standard benchmark for abstractive summarization.
* **Content:** Contains news articles (average ~700-800 words) paired with multi-sentence highlights/summaries (average ~60-70 words) written by journalists.
* **Size:** ~287k training, ~13k validation, ~11k test samples.
* **Task:** The goal is to generate a concise, abstractive summary capturing the main points of the article.
* **Source:** `abisee/cnn_dailymail` (version 3.0.0) from the Hugging Face Hub.

## 💡 Model: TinyLlama-1.1B

**TinyLlama-1.1B-Chat-v1.0** was chosen for its excellent balance between size, performance, and resource requirements.
* **Architecture:** A compact, instruction-tuned Llama-based decoder-only transformer.
* **Size:** 1.1 billion parameters.
* **Training:** Pre-trained on a large corpus (~3 trillion tokens).
* **Efficiency:** Suitable for fine-tuning and inference on single GPUs with moderate VRAM (compatible with 4-bit quantization).
* **Rationale:** Ideal for experimentation and demonstrating efficient fine-tuning techniques in environments like Google Colab or Kaggle.

## 🛠️ Methodology

The project follows these key steps as implemented in the notebook:

1.  **Setup:** Installs necessary libraries.
2.  **Environment Configuration:** Sets up PyTorch device (GPU/CPU) and initializes Ray for distributed tuning.
3.  **Data Loading:** Loads the CNN/DailyMail dataset using the `datasets` library.
4.  **Preprocessing & Tokenization:**
    * Formats article-summary pairs into a prompt structure (`"Summarize: {article}\n\nSummary: {summary}"`).
    * Tokenizes the prompts using the TinyLlama tokenizer, applying padding and truncation (`max_length=256`).
5.  **Hyperparameter Optimization (Ray Tune):**
    * Defines a search space for `learning_rate`, `gradient_accumulation_steps`, and `weight_decay`.
    * Uses a training function (`train_tinyllama_ray_with_rouge`) that incorporates 4-bit quantization and LoRA.
    * Each trial trains for a small number of steps (`max_steps=25`) and reports validation loss and ROUGE scores (calculated on a sample).
    * Ray Tune identifies the best configuration based on minimizing validation loss and maximizing ROUGE.
6.  **Model Training (Best vs. Baseline):**
    * Trains a model using the **best hyperparameters** found by Ray Tune.
    * Trains a **baseline model** using default hyperparameters.
    * Both models use 4-bit quantization and LoRA for consistency. Training is done for a limited number of steps (`max_steps=25`) for demonstration.
7.  **Evaluation:**
    * Calculates final evaluation loss and perplexity for both models.
    * Computes ROUGE-1, ROUGE-2, and ROUGE-L scores on the **test set** (using a sample for efficiency) for both models.
8.  **Results & Visualization:**
    * Presents a comparison table (`pandas.DataFrame`) summarizing the performance metrics.
    * Generates plots comparing the optimized model vs. baseline and against SOTA ROUGE scores (BART-Large, PEGASUS).
9.  **Error Analysis:** Qualitatively analyzes summaries generated by the best model on test samples to identify common error types (length issues, coherence).
10. **Sample Prediction:** Uses the fine-tuned (best) model to generate a summary for a new, provided article text.

## ⚙️ Setup and Installation

1.  **Clone Repository (Optional):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create Virtual Environment:** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # .\venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Use the `requirements.txt` file provided above)*

## 💻 Environment

* **Python:** 3.11+ recommended.
* **Key Libraries:** `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `ray[tune]`, `rouge-score`, `pandas`, `numpy`, `matplotlib`, `seaborn`.
* **Hardware:**
    * **GPU:** Strongly recommended. A GPU with CUDA support and >= 12-16GB VRAM is ideal for running the 4-bit quantized fine-tuning (e.g., NVIDIA T4, P100, V100, A100 or consumer equivalents).
    * **CPU:** Possible but likely prohibitively slow for training.
* **Platform:** Developed and tested in environments like Google Colab/Kaggle. Ensure necessary GPU runtimes are enabled if using these platforms.

## ▶️ How to Run

1.  **Ensure Environment is Set Up:** Activate your virtual environment and confirm all packages from `requirements.txt` are installed.
2.  **Launch Jupyter:** Start Jupyter Lab or Jupyter Notebook.
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
3.  **Open the Notebook:** Navigate to and open `FineTuneTinyLlama.ipynb`.
4.  **Run Cells Sequentially:** Execute the notebook cells in order. Pay attention to:
    * **GPU Availability:** The notebook checks for and utilizes CUDA if available. Ensure your runtime has GPU access enabled.
    * **Ray Initialization:** Cells initialize the Ray runtime.
    * **Dataset Download:** The first time running, the CNN/DailyMail dataset will be downloaded (this may take some time).
    * **Model Download:** The TinyLlama model weights will be downloaded from Hugging Face Hub.
    * **Hyperparameter Tuning:** The Ray Tune section will run multiple short training trials. This is compute-intensive and will take the longest.
    * **Final Training:** Two short training runs (best and baseline) are performed.
    * **Evaluation & Analysis:** ROUGE scores and error analysis are performed on samples.
5.  **Review Outputs:** Examine the printed tables, visualizations, error analysis, and the final sample prediction. The comparison results are also saved to `model_comparison_results.csv`.

## 📈 Results

The notebook generates:
* A results table comparing the hyperparameters and performance (Loss, Perplexity, ROUGE scores, Training Time) of the baseline and optimized models.
* Visualizations comparing ROUGE scores, performance metrics, and benchmarking against SOTA models.
* Qualitative error analysis output and sample summary comparisons.
* The final trained (best) model artifacts saved to the `./final_best_model` directory.

## ✍️ Sample Prediction

The final cells demonstrate how to load the fine-tuned model and use it to generate a summary for an arbitrary piece of text, showcasing its practical application.

## ✅ Conclusion

This project successfully fine-tunes TinyLlama-1.1B for summarization using efficient techniques. The results demonstrate that even smaller models, when combined with quantization, PEFT (LoRA), and automated hyperparameter optimization (Ray Tune), can achieve competitive performance on complex tasks like abstractive summarization, offering a viable path for adapting LLMs in resource-limited settings. The optimized model shows clear improvements over the baseline and achieves a significant fraction of SOTA performance with minimal training.