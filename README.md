This project implements a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs using the popular Kaggle dataset. The dataset contains 25,000 labeled training images and 12,500 test images.

📂 Dataset

Source: Kaggle Dogs vs. Cats Dataset

Train set: 25,000 images (12,500 cats + 12,500 dogs)

Test set: 12,500 images (unlabeled)

Each training image is named as:

cat.0.jpg, cat.1.jpg, …

dog.0.jpg, dog.1.jpg, …

⚙️ Preprocessing

Resize images (e.g., 128×128).

Convert to grayscale (optional).

Extract features:

HOG (Histogram of Oriented Gradients) or

Pre-trained CNN embeddings (e.g., VGG16)

Normalize features using StandardScaler.

🚀 Model

Classifier: Support Vector Machine (SVM)

Kernel: linear or rbf

Libraries: scikit-learn, opencv, matplotlib

📊 Results

Evaluation Metrics: Accuracy, Precision, Recall, F1-score.

Confusion Matrix for class distribution.

Expected accuracy: ~70–80% with HOG features; higher (~90%+) with CNN feature extraction + SVM.

📌 Project Structure
dogs-vs-cats-svm/
│-- data/
│   ├── train/
│   │    ├── cat.0.jpg
│   │    ├── dog.0.jpg
│   └── test1/
│-- notebooks/
│   └── svm_classification.ipynb
│-- src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train_svm.py
│-- results/
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│-- README.md
│-- requirements.txt

📦 Installation
git clone https://github.com/your-username/dogs-vs-cats-svm.git
cd dogs-vs-cats-svm
pip install -r requirements.txt

▶️ Usage

Place dataset in the data/ folder.

Run preprocessing & feature extraction:

python src/preprocess.py


Train the SVM model:

python src/train_svm.py


Evaluate results:

jupyter notebook notebooks/svm_classification.ipynb

🛠️ Requirements

Python 3.8+

scikit-learn

OpenCV

NumPy

Matplotlib

Jupyter Notebook

📌 Future Improvements

Try deep learning (CNN) for higher accuracy.

Hyperparameter tuning with GridSearchCV.

Deploy as a Flask/Django web app.

👨‍💻 Author

Shubham Maurya

B.Tech CSE (AI/ML), SRMCEM

LinkedIn
