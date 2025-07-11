#!/usr/bin/env python3
"""
htypist - A command-line typing game for practicing Python code

A terminal-based typing game that presents Python code snippets for users to type,
providing real-time feedback and calculating typing speed and accuracy.
"""

import os
import time
import random


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class CodeSnippets:
    """Collection of Python code snippets for typing practice"""

    SNIPPETS = [
        # Basic Python snippets
        "def hello_world():\n    print('Hello, World!')\n    return True",

        "import os\nfor file in os.listdir('.'):\n    if file.endswith('.py'):\n        print(f'Python file: {file}')",

        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",

        "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n    \n    def __str__(self):\n        return f'{self.name}, {self.age} years old'",

        "try:\n    result = 10 / 0\nexcept ZeroDivisionError as e:\n    print(f'Error: {e}')\nfinally:\n    print('Cleanup completed')",

        "numbers = [1, 2, 3, 4, 5]\nsquares = [x**2 for x in numbers if x % 2 == 0]\nprint(f'Even squares: {squares}')",

        "with open('data.txt', 'r') as file:\n    lines = file.readlines()\n    for i, line in enumerate(lines, 1):\n        print(f'Line {i}: {line.strip()}')",

        "def decorator(func):\n    def wrapper(*args, **kwargs):\n        print(f'Calling {func.__name__}')\n        return func(*args, **kwargs)\n    return wrapper",

        "import json\ndata = {'name': 'John', 'age': 30, 'city': 'New York'}\njson_string = json.dumps(data, indent=2)\nprint(json_string)",

        # PyTorch snippets
        "import torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super(SimpleNet, self).__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(hidden_size, output_size)",

        "import torch\nimport torch.optim as optim\n\nmodel = SimpleNet(784, 128, 10)\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\n\nfor epoch in range(num_epochs):\n    optimizer.zero_grad()\n    outputs = model(inputs)\n    loss = criterion(outputs, targets)\n    loss.backward()\n    optimizer.step()",

        "import torch\ntensor_a = torch.randn(3, 4)\ntensor_b = torch.randn(4, 5)\nresult = torch.matmul(tensor_a, tensor_b)\nprint(f'Shape: {result.shape}')\nif torch.cuda.is_available():\n    device = torch.device('cuda')\n    tensor_gpu = tensor_a.to(device)",

        "from torch.utils.data import DataLoader, Dataset\n\nclass CustomDataset(Dataset):\n    def __init__(self, data, labels):\n        self.data = data\n        self.labels = labels\n    \n    def __len__(self):\n        return len(self.data)\n    \n    def __getitem__(self, idx):\n        return self.data[idx], self.labels[idx]",

        # Scikit-learn snippets
        "from sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nrf_model = RandomForestClassifier(n_estimators=100, random_state=42)\nrf_model.fit(X_train, y_train)\ny_pred = rf_model.predict(X_test)\naccuracy = accuracy_score(y_test, y_pred)",

        "from sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.pipeline import Pipeline\n\nscaler = StandardScaler()\nencoder = LabelEncoder()\nX_scaled = scaler.fit_transform(X)\ny_encoded = encoder.fit_transform(y)\n\npipeline = Pipeline([\n    ('scaler', StandardScaler()),\n    ('classifier', RandomForestClassifier())\n])\npipeline.fit(X_train, y_train)",

        "from sklearn.model_selection import GridSearchCV\nfrom sklearn.svm import SVC\n\nparam_grid = {\n    'C': [0.1, 1, 10, 100],\n    'gamma': ['scale', 'auto'],\n    'kernel': ['rbf', 'linear']\n}\nsvm_model = SVC()\ngrid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')\ngrid_search.fit(X_train, y_train)\nprint(f'Best params: {grid_search.best_params_}')",

        "import numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n\nkmeans = KMeans(n_clusters=3, random_state=42)\ncluster_labels = kmeans.fit_predict(X)\nsilhouette_avg = silhouette_score(X, cluster_labels)\nprint(f'Silhouette Score: {silhouette_avg:.3f}')\ncenters = kmeans.cluster_centers_",

        # TensorFlow/Keras snippets
        "import tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Dropout\n\nmodel = Sequential([\n    Dense(128, activation='relu', input_shape=(784,)),\n    Dropout(0.2),\n    Dense(64, activation='relu'),\n    Dense(10, activation='softmax')\n])\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])",

        "import tensorflow as tf\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\n\ndatagen = ImageDataGenerator(\n    rotation_range=20,\n    width_shift_range=0.2,\n    height_shift_range=0.2,\n    horizontal_flip=True,\n    validation_split=0.2\n)\ntrain_generator = datagen.flow_from_directory(\n    'data/train',\n    target_size=(224, 224),\n    batch_size=32,\n    subset='training'\n)",

        "import tensorflow as tf\n\n@tf.function\ndef train_step(model, optimizer, x_batch, y_batch):\n    with tf.GradientTape() as tape:\n        predictions = model(x_batch, training=True)\n        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)\n    gradients = tape.gradient(loss, model.trainable_variables)\n    optimizer.apply_gradients(zip(gradients, model.trainable_variables))\n    return loss",

        # Data Science & ML snippets
        "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\ndf = pd.read_csv('data.csv')\nprint(df.head())\nprint(df.info())\nprint(df.describe())\nmissing_data = df.isnull().sum()\ndf_cleaned = df.dropna()\ncorrelation_matrix = df.corr()\nplt.figure(figsize=(10, 8))\nplt.heatmap(correlation_matrix, annot=True)",

        "import seaborn as sns\nimport matplotlib.pyplot as plt\nfrom scipy import stats\n\nplt.figure(figsize=(12, 8))\nplt.subplot(2, 2, 1)\nsns.histplot(data=df, x='feature1', kde=True)\nplt.subplot(2, 2, 2)\nsns.boxplot(data=df, x='category', y='value')\nplt.subplot(2, 2, 3)\nsns.scatterplot(data=df, x='feature1', y='feature2', hue='target')\nplt.tight_layout()\nplt.show()",

        "from sklearn.feature_selection import SelectKBest, f_classif\nfrom sklearn.decomposition import PCA\n\n# Feature selection\nselector = SelectKBest(score_func=f_classif, k=10)\nX_selected = selector.fit_transform(X, y)\nselected_features = selector.get_support(indices=True)\n\n# Dimensionality reduction\npca = PCA(n_components=0.95)\nX_pca = pca.fit_transform(X_scaled)\nprint(f'Original features: {X.shape[1]}, PCA features: {X_pca.shape[1]}')",

        # Google Cloud snippets
        "from google.cloud import storage\nfrom google.cloud import bigquery\n\n# Cloud Storage\nclient = storage.Client()\nbucket = client.bucket('my-bucket')\nblob = bucket.blob('data/file.csv')\nblob.upload_from_filename('local_file.csv')\nprint(f'File uploaded to {blob.name}')\n\n# BigQuery\nbq_client = bigquery.Client()\nquery = 'SELECT * FROM `project.dataset.table` LIMIT 100'\nresults = bq_client.query(query).to_dataframe()",

        "from google.cloud import aiplatform\nfrom google.cloud.aiplatform import gapic as aip\n\naiplatform.init(project='my-project', location='us-central1')\n\n# Deploy model\nmodel = aiplatform.Model.upload(\n    display_name='my-model',\n    artifact_uri='gs://my-bucket/model/',\n    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest'\n)\nendpoint = model.deploy(machine_type='n1-standard-2')",

        "from google.cloud import secretmanager\nfrom google.oauth2 import service_account\nimport os\n\nclient = secretmanager.SecretManagerServiceClient()\nname = f'projects/{project_id}/secrets/{secret_id}/versions/latest'\nresponse = client.access_secret_version(request={'name': name})\nsecret_value = response.payload.data.decode('UTF-8')\nos.environ['API_KEY'] = secret_value",

        # Advanced ML snippets
        "import optuna\nfrom sklearn.ensemble import GradientBoostingClassifier\n\ndef objective(trial):\n    n_estimators = trial.suggest_int('n_estimators', 50, 300)\n    max_depth = trial.suggest_int('max_depth', 3, 10)\n    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)\n    \n    model = GradientBoostingClassifier(\n        n_estimators=n_estimators,\n        max_depth=max_depth,\n        learning_rate=learning_rate\n    )\n    model.fit(X_train, y_train)\n    return accuracy_score(y_val, model.predict(X_val))",

        "import mlflow\nimport mlflow.sklearn\n\nwith mlflow.start_run():\n    mlflow.log_param('n_estimators', 100)\n    mlflow.log_param('max_depth', 5)\n    \n    model = RandomForestClassifier(n_estimators=100, max_depth=5)\n    model.fit(X_train, y_train)\n    \n    accuracy = accuracy_score(y_test, model.predict(X_test))\n    mlflow.log_metric('accuracy', accuracy)\n    mlflow.sklearn.log_model(model, 'random_forest_model')",

        "from transformers import AutoTokenizer, AutoModel\nimport torch\n\ntokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\nmodel = AutoModel.from_pretrained('bert-base-uncased')\n\ninputs = tokenizer('Hello, world!', return_tensors='pt')\nwith torch.no_grad():\n    outputs = model(**inputs)\nlast_hidden_states = outputs.last_hidden_state\nprint(f'Output shape: {last_hidden_states.shape}')"
    ]

    @classmethod
    def get_random_snippet(cls) -> str:
        """Get a random code snippet"""
        return random.choice(cls.SNIPPETS)


class TypingStats:
    """Calculate and track typing statistics"""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.total_chars = 0
        self.correct_chars = 0
        self.errors = 0

    def start_typing(self):
        """Mark the start of typing"""
        self.start_time = time.time()

    def end_typing(self):
        """Mark the end of typing"""
        self.end_time = time.time()

    def add_character(self, is_correct: bool):
        """Add a typed character to statistics"""
        self.total_chars += 1
        if is_correct:
            self.correct_chars += 1
        else:
            self.errors += 1

    def get_wpm(self) -> float:
        """Calculate Words Per Minute (assuming 5 characters per word)"""
        if self.start_time == 0.0 or self.end_time == 0.0:
            return 0.0

        time_minutes = (self.end_time - self.start_time) / 60
        if time_minutes == 0:
            return 0.0

        words = self.correct_chars / 5  # Standard: 5 characters = 1 word
        return words / time_minutes

    def get_accuracy(self) -> float:
        """Calculate accuracy percentage"""
        if self.total_chars == 0:
            return 0.0
        return (self.correct_chars / self.total_chars) * 100


class TypingGame:
    """Main typing game class"""

    def __init__(self):
        self.current_snippet = ""
        self.user_input = ""
        self.position = 0
        self.stats = TypingStats()
        self.is_finished = False
        self.user_typed_content = ""
        self.target_content = ""

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def display_welcome(self):
        """Display welcome message"""
        self.clear_screen()
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                         HTYPIST                              ║")
        print("║                Python Code Typing Practice                   ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}")
        print(f"{Colors.YELLOW}Instructions:{Colors.RESET}")
        print("• Type the Python code exactly as shown")
        print("• Indentation will be provided automatically")
        print("• Green text indicates correct typing")
        print("• Red text indicates errors")
        print("• Press Ctrl+C to quit")
        print()
        input(f"{Colors.BLUE}Press Enter to start...{Colors.RESET}")

    def display_snippet_with_progress(self):
        """Display the code snippet with typing progress"""
        self.clear_screen()
        print(f"{Colors.BOLD}Type the following Python code:{Colors.RESET}")
        print("─" * 60)        # Display the target text with color coding (content only, no auto-indentation)
        target_lines = self.current_snippet.split('\n')
        for line in target_lines:
            # Show indentation in blue, content in white
            indentation = len(line) - len(line.lstrip())
            indent_str = ' ' * indentation
            content = line.lstrip()

            print(f"{Colors.BLUE}{indent_str}{Colors.WHITE}{content}{Colors.RESET}")

        print("─" * 60)
        print(f"{Colors.BLUE}Your input:{Colors.RESET}")

        # Display user input with color coding based on content comparison
        if hasattr(self, 'user_typed_content') and hasattr(self, 'target_content'):
            user_lines = self.user_input.split('\n') if self.user_input else []
            target_lines = self.current_snippet.split('\n')

            for user_line, target_line in zip(user_lines, target_lines):
                # Show indentation in blue
                target_indentation = len(target_line) - len(target_line.lstrip())
                indent_str = ' ' * target_indentation
                print(f"{Colors.BLUE}{indent_str}{Colors.RESET}", end="")

                # Get content without indentation for both
                user_content = user_line.lstrip() if user_line else ""
                target_content = target_line.lstrip()

                # Use edit distance to find optimal alignment and show precise errors
                def get_alignment(s1, s2):
                    """Get character-level alignment using edit distance traceback"""
                    m, n = len(s1), len(s2)
                    dp = [[0] * (n + 1) for _ in range(m + 1)]

                    # Fill DP table
                    for i in range(m + 1):
                        dp[i][0] = i
                    for j in range(n + 1):
                        dp[0][j] = j

                    for i in range(1, m + 1):
                        for j in range(1, n + 1):
                            if s1[i-1] == s2[j-1]:
                                dp[i][j] = dp[i-1][j-1]
                            else:
                                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

                    # Traceback to get alignment
                    alignment = []
                    i, j = m, n
                    while i > 0 or j > 0:
                        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
                            alignment.append(('match', s1[i-1], s2[j-1]))
                            i, j = i-1, j-1
                        elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
                            alignment.append(('delete', s1[i-1], None))
                            i -= 1
                        else:
                            alignment.append(('insert', None, s2[j-1]))
                            j -= 1

                    return list(reversed(alignment))

                # Get alignment and display with precise coloring
                if user_content or target_content:
                    alignment = get_alignment(user_content, target_content)

                    for op, user_char, target_char in alignment:
                        if op == 'match':
                            print(f"{Colors.GREEN}{user_char}{Colors.RESET}", end="")
                        elif op == 'delete':
                            # User typed extra character - show in red
                            print(f"{Colors.RED}{user_char}{Colors.RESET}", end="")
                        elif op == 'insert':
                            # User missed character - show missing char in red
                            print(f"{Colors.RED}{target_char}{Colors.RESET}", end="")

                print()  # New line

        # Show current statistics
        if self.stats.total_chars > 0:
            accuracy = self.stats.get_accuracy()
            target_content_only = '\n'.join([line.lstrip() for line in self.current_snippet.split('\n')])
            user_content_only = getattr(self, 'user_typed_content', '')
            print(f"\n{Colors.YELLOW}Progress: {len(user_content_only)}/{len(target_content_only)} characters | "
                  f"Accuracy: {accuracy:.1f}%{Colors.RESET}")

    def play_round(self):
        """Play a single typing round using multi-line input method"""
        self.current_snippet = CodeSnippets.get_random_snippet()
        self.user_input = ""
        self.position = 0
        self.stats = TypingStats()
        self.is_finished = False

        self.display_snippet_with_progress()
        self.stats.start_typing()

        try:
            print(f"\n{Colors.CYAN}Type the code above (indentation will be provided automatically):{Colors.RESET}")
            print(f"{Colors.YELLOW}Press Ctrl+C to quit, Ctrl+D to skip remaining lines{Colors.RESET}\n")

            lines = []
            user_typed_lines = []  # Track only what user actually typed
            snippet_lines = self.current_snippet.split('\n')

            for target_line in snippet_lines:
                try:
                    # Extract indentation from the target line
                    indentation = len(target_line) - len(target_line.lstrip())
                    indent_str = ' ' * indentation
                    target_content = target_line.lstrip()  # Content without indentation

                    # Show the indentation and let user type the rest
                    print(f"{Colors.BLUE}{indent_str}{Colors.RESET}", end="", flush=True)

                    # Get user input for the rest of the line
                    user_line_content = input()

                    # Store what user actually typed (without auto-indentation)
                    user_typed_lines.append(user_line_content)

                    # Combine indentation with user input for display
                    full_line = indent_str + user_line_content
                    lines.append(full_line)

                except EOFError:
                    # User pressed Ctrl+D
                    break

            # Join lines with newlines to reconstruct the full input
            self.user_input = '\n'.join(lines)
            self.stats.end_typing()

            # Calculate statistics based on what user actually typed vs target content
            target_content_lines = [line.lstrip() for line in self.current_snippet.split('\n')]
            user_typed_content = '\n'.join(user_typed_lines)
            target_content = '\n'.join(target_content_lines)

            # Store for display purposes
            self.user_typed_content = user_typed_content
            self.target_content = target_content

            # Calculate statistics using edit distance for more accurate error counting
            def calculate_edit_distance(s1, s2):
                """Calculate minimum edit distance (Levenshtein distance)"""
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]

                # Initialize base cases
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j

                # Fill the DP table
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1]  # No operation needed
                        else:
                            dp[i][j] = 1 + min(
                                dp[i-1][j],    # Deletion
                                dp[i][j-1],    # Insertion
                                dp[i-1][j-1]   # Substitution
                            )

                return dp[m][n]

            # Calculate accurate statistics
            errors = calculate_edit_distance(user_typed_content, target_content)
            correct_chars = max(0, len(user_typed_content) - errors)
            total_chars = len(user_typed_content)

            # Update stats with accurate counts
            self.stats.total_chars = total_chars
            self.stats.correct_chars = correct_chars
            self.stats.errors = errors

            self.display_snippet_with_progress()
            return True

        except KeyboardInterrupt:
            return False

    def display_results(self):
        """Display results after completing a round"""
        print(f"\n{Colors.BOLD}{Colors.GREEN}Round Complete!{Colors.RESET}")
        print("═" * 50)

        wpm = self.stats.get_wpm()
        accuracy = self.stats.get_accuracy()
        time_taken = self.stats.end_time - self.stats.start_time

        print(f"{Colors.CYAN}Statistics:{Colors.RESET}")
        print(f"  Time taken: {Colors.YELLOW}{time_taken:.2f} seconds{Colors.RESET}")
        print(f"  Speed: {Colors.YELLOW}{wpm:.1f} WPM{Colors.RESET}")
        print(f"  Accuracy: {Colors.YELLOW}{accuracy:.1f}%{Colors.RESET}")
        print(f"  Characters typed: {Colors.YELLOW}{self.stats.total_chars}{Colors.RESET}")
        print(f"  Errors: {Colors.YELLOW}{self.stats.errors}{Colors.RESET}")

        # Performance feedback
        if accuracy >= 95 and wpm >= 40:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Excellent! You're a Python typing master!{Colors.RESET}")
        elif accuracy >= 90 and wpm >= 30:
            print(f"\n{Colors.GREEN}👍 Great job! Keep practicing to improve further.{Colors.RESET}")
        elif accuracy >= 80:
            print(f"\n{Colors.YELLOW}👌 Good work! Focus on accuracy before speed.{Colors.RESET}")
        else:
            print(f"\n{Colors.BLUE}💪 Keep practicing! Accuracy will improve with time.{Colors.RESET}")

    def play(self):
        """Main game loop"""
        try:
            self.display_welcome()

            while True:
                if not self.play_round():
                    break

                self.display_results()

                # Ask to play again
                print(f"\n{Colors.BLUE}Would you like to practice with another snippet? (y/n): {Colors.RESET}", end="")
                choice = input().lower().strip()

                if choice not in ['y', 'yes']:
                    break

            print(f"\n{Colors.CYAN}Thanks for practicing with htypist! Keep coding! 🐍{Colors.RESET}")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Thanks for practicing with htypist! Keep coding! 🐍{Colors.RESET}")


def main():
    """Main function"""
    game = TypingGame()
    game.play()


if __name__ == "__main__":
    main()
