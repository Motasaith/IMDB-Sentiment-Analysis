"""
IMDb Sentiment Analysis - Machine Learning Models
Phase 1: Model Building and Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support,
                           roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

class SentimentModels:
    """Class to handle multiple sentiment analysis models"""
    
    def __init__(self):
        """Initialize the models and vectorizers"""
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        self.best_model = None
        self.best_vectorizer = None
        
    def create_models(self):
        """Initialize different ML models"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        self.vectorizers = {
            'tfidf': TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2)),
            'count': CountVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
        }
        
        print("✅ Models and vectorizers initialized!")
        print(f"📊 Models: {list(self.models.keys())}")
        print(f"📊 Vectorizers: {list(self.vectorizers.keys())}")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all model combinations"""
        print("🚀 Starting model training and evaluation...")
        
        self.results = {}
        
        for vec_name, vectorizer in self.vectorizers.items():
            print(f"\\n{'='*60}")
            print(f"🔄 Processing with {vec_name.upper()} Vectorizer")
            print('='*60)
            
            # Fit vectorizer and transform data
            print(f"📊 Fitting {vec_name} vectorizer...")
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            print(f"✅ Vector shape - Train: {X_train_vec.shape}, Test: {X_test_vec.shape}")
            
            for model_name, model in self.models.items():
                print(f"\\n🤖 Training {model_name}...")
                
                # Train model
                model.fit(X_train_vec, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_vec)
                y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                key = f"{vec_name}_{model_name}"
                self.results[key] = {
                    'model': model,
                    'vectorizer': vectorizer,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"✅ {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        # Find best model
        self.find_best_model()
        
        # Create evaluation plots
        self.create_evaluation_plots(y_test)
        
        return self.results
    
    def find_best_model(self):
        """Find the best performing model"""
        if not self.results:
            print("❌ No results available!")
            return
            
        # Sort by F1 score
        best_key = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        best_result = self.results[best_key]
        
        self.best_model = best_result['model']
        self.best_vectorizer = best_result['vectorizer']
        
        print(f"\\n🏆 BEST MODEL: {best_key}")
        print("="*50)
        print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
        print(f"📊 Precision: {best_result['precision']:.4f}")
        print(f"📊 Recall: {best_result['recall']:.4f}")
        print(f"📊 F1-Score: {best_result['f1']:.4f}")
        print(f"📊 AUC: {best_result['auc']:.4f}")
        
        return best_key, best_result
    
    def create_evaluation_plots(self, y_test, save_plots=True):
        """Create comprehensive evaluation visualizations"""
        if not self.results:
            print("❌ No results to plot!")
            return
            
        print("📊 Creating evaluation plots...")
        
        # Prepare data for plotting
        model_names = []
        accuracies = []
        f1_scores = []
        aucs = []
        
        for key, result in self.results.items():
            model_names.append(key.replace('_', '\\n'))
            accuracies.append(result['accuracy'])
            f1_scores.append(result['f1'])
            aucs.append(result['auc'])
        
        # Create plots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Model Comparison - Accuracy
        plt.subplot(2, 3, 1)
        bars1 = plt.bar(range(len(model_names)), accuracies, color='lightblue', edgecolor='black')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0.8, 1.0)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Model Comparison - F1 Score
        plt.subplot(2, 3, 2)
        bars2 = plt.bar(range(len(model_names)), f1_scores, color='lightgreen', edgecolor='black')
        plt.xlabel('Models')
        plt.ylabel('F1 Score')
        plt.title('Model F1 Score Comparison')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0.8, 1.0)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # 3. Model Comparison - AUC
        plt.subplot(2, 3, 3)
        bars3 = plt.bar(range(len(model_names)), aucs, color='lightcoral', edgecolor='black')
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model AUC Score Comparison')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0.8, 1.0)
        
        # Add value labels on bars
        for bar, auc in zip(bars3, aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # 4. Confusion Matrix for Best Model
        plt.subplot(2, 3, 4)
        best_key = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        best_y_pred = self.results[best_key]['y_pred']
        cm = confusion_matrix(y_test, best_y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix\\n{best_key}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 5. ROC Curves
        plt.subplot(2, 3, 5)
        for key, result in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{key} (AUC: {result['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 6. Metrics Heatmap
        plt.subplot(2, 3, 6)
        metrics_data = []
        for key, result in self.results.items():
            metrics_data.append([
                result['accuracy'],
                result['precision'],
                result['recall'],
                result['f1'],
                result['auc']
            ])
        
        metrics_df = pd.DataFrame(metrics_data,
                                columns=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                                index=[key.replace('_', '\\n') for key in self.results.keys()])
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Score'})
        plt.title('All Metrics Heatmap')
        plt.ylabel('Models')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detailed_evaluation(self, y_test):
        """Print detailed evaluation for all models"""
        if not self.results:
            print("❌ No results available!")
            return
            
        print("\\n" + "="*80)
        print("📊 DETAILED MODEL EVALUATION")
        print("="*80)
        
        for key, result in self.results.items():
            print(f"\\n🤖 {key.upper()}")
            print("-" * 50)
            print(classification_report(y_test, result['y_pred'], 
                                      target_names=['Negative', 'Positive']))
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='logistic_regression', vec_name='tfidf'):
        """Perform hyperparameter tuning for specified model"""
        print(f"🔧 Hyperparameter tuning for {model_name} with {vec_name}...")
        
        # Get vectorizer and fit
        vectorizer = self.vectorizers[vec_name]
        X_train_vec = vectorizer.fit_transform(X_train)
        
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        }
        
        if model_name not in param_grids:
            print(f"❌ No parameter grid defined for {model_name}")
            return None
            
        # Perform grid search
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=3, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_vec, y_train)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_models(self, save_dir="models"):
        """Save trained models and vectorizers"""
        import os
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if not self.results:
            print("❌ No models to save!")
            return
            
        print(f"💾 Saving models to {save_dir}/...")
        
        # Save best model and vectorizer
        if self.best_model and self.best_vectorizer:
            joblib.dump(self.best_model, f"{save_dir}/best_model.pkl")
            joblib.dump(self.best_vectorizer, f"{save_dir}/best_vectorizer.pkl")
            print("✅ Best model and vectorizer saved!")
        
        # Save all results
        results_to_save = {}
        for key, result in self.results.items():
            results_to_save[key] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'auc': result['auc']
            }
            # Save individual models
            joblib.dump(result['model'], f"{save_dir}/{key}_model.pkl")
            joblib.dump(result['vectorizer'], f"{save_dir}/{key}_vectorizer.pkl")
        
        # Save results summary
        results_df = pd.DataFrame(results_to_save).T
        results_df.to_csv(f"{save_dir}/model_results.csv")
        
        print(f"✅ All models saved in {save_dir}/")
        print(f"📁 Files created:")
        for file in os.listdir(save_dir):
            print(f"  - {file}")
    
    def predict_sentiment(self, text, use_best_model=True):
        """Predict sentiment for a single text"""
        if use_best_model:
            if not self.best_model or not self.best_vectorizer:
                print("❌ Best model not available!")
                return None
            model = self.best_model
            vectorizer = self.best_vectorizer
        else:
            # Use first available model
            first_key = list(self.results.keys())[0]
            model = self.results[first_key]['model']
            vectorizer = self.results[first_key]['vectorizer']
        
        # Preprocess and vectorize
        text_vec = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': probability[0],
                'positive': probability[1]
            }
        }

def main():
    """Main function to run the ML pipeline"""
    print("🚀 Starting Machine Learning Pipeline...")
    
    # This would typically load preprocessed data
    # For now, we'll assume data is loaded from preprocessing
    from data_preprocessing import IMDbDataProcessor
    
    # Load and preprocess data
    processor = IMDbDataProcessor()
    processor.load_data()
    processor.preprocess_data()
    X_train, X_test, y_train, y_test = processor.prepare_for_modeling()
    
    # Initialize and train models
    sentiment_models = SentimentModels()
    sentiment_models.create_models()
    
    # Train and evaluate
    results = sentiment_models.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Detailed evaluation
    sentiment_models.detailed_evaluation(y_test)
    
    # Save models
    sentiment_models.save_models()
    
    # Test prediction
    test_review = "This movie was absolutely fantastic! Great acting and storyline."
    result = sentiment_models.predict_sentiment(test_review)
    print(f"\\n🧪 Test prediction:")
    print(f"Review: {test_review}")
    print(f"Predicted: {result['sentiment']} (confidence: {result['confidence']:.3f})")
    
    print("\\n🎉 Machine Learning pipeline completed successfully!")
    
    return sentiment_models, results

if __name__ == "__main__":
    sentiment_models, results = main()
