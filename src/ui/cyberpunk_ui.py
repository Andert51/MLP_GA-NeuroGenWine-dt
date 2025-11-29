"""
VinoGen-CyberCore: Cyberpunk Terminal UI
Matrix-style interface with rich library for stunning CLI experience.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.box import DOUBLE, ROUNDED, HEAVY
from rich import print as rprint
from rich.markdown import Markdown
import time
from typing import Dict, List, Optional
import pyfiglet
from datetime import datetime


class CyberpunkUI:
    """
    Cyberpunk/Matrix-themed Terminal User Interface.
    
    Features:
    - Boot sequences
    - Real-time dashboards
    - Progress bars with neon styling
    - ASCII art headers
    - Verbose logging with color coding
    - Data tables
    - System status panels
    """
    
    def __init__(self):
        """Initialize the Cyberpunk Terminal."""
        self.console = Console()
        
        # Cyberpunk color scheme
        self.colors = {
            'neon_green': '#39FF14',
            'electric_blue': '#00FFFF',
            'deep_purple': '#9D00FF',
            'hot_pink': '#FF10F0',
            'cyber_yellow': '#FFFF00',
            'warning': '#FF6600',
            'error': '#FF0000',
            'success': '#00FF00',
            'info': '#00BFFF'
        }
    
    def clear_screen(self):
        """Clear the terminal."""
        self.console.clear()
    
    def show_boot_sequence(self):
        """Display cyberpunk boot sequence."""
        self.clear_screen()
        
        # ASCII Banner
        banner = pyfiglet.figlet_format("NEUROGEN", font="slant")
        subtitle = pyfiglet.figlet_format("WineLab", font="digital")
        
        self.console.print(banner, style=f"bold {self.colors['neon_green']}")
        self.console.print(subtitle, style=f"bold {self.colors['electric_blue']}")
        self.console.print()
        
        # Boot messages
        boot_messages = [
            ("[SYSTEM]", "Initializing Neural Core...", self.colors['info']),
            ("[QUANTUM]", "Loading Genetic Algorithm Engine...", self.colors['electric_blue']),
            ("[DATA]", "Establishing Data Pipeline...", self.colors['neon_green']),
            ("[CYBER]", "Activating Visualization Engine...", self.colors['deep_purple']),
            ("[NEURAL]", "Calibrating Synaptic Weights...", self.colors['hot_pink']),
            ("[AI]", "Deploying Evolution Protocol...", self.colors['cyber_yellow']),
            ("[READY]", "NeuroGen WineLab Online!", self.colors['success'])
        ]
        
        with Progress(
            SpinnerColumn(spinner_name="dots", style=self.colors['electric_blue']),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            for prefix, message, color in boot_messages:
                task = progress.add_task(f"[{color}]{prefix}[/] {message}", total=1)
                time.sleep(0.5)
                progress.update(task, completed=1)
        
        self.console.print()
        self.show_divider()
    
    def show_divider(self, char: str = "═", width: int = 80, color: str = None):
        """Print a divider line."""
        color = color or self.colors['electric_blue']
        self.console.print(char * width, style=color)
    
    def show_header(self, title: str, subtitle: str = ""):
        """Display section header."""
        self.console.print()
        panel = Panel(
            f"[bold {self.colors['neon_green']}]{title}[/]\n"
            f"[{self.colors['electric_blue']}]{subtitle}[/]",
            border_style=self.colors['deep_purple'],
            box=DOUBLE,
            expand=False
        )
        self.console.print(Align.center(panel))
        self.console.print()
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with color-coded level.
        
        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, SUCCESS, DEBUG)
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        level_colors = {
            "INFO": self.colors['info'],
            "WARNING": self.colors['warning'],
            "ERROR": self.colors['error'],
            "SUCCESS": self.colors['success'],
            "DEBUG": self.colors['deep_purple'],
            "SYSTEM": self.colors['cyber_yellow']
        }
        
        color = level_colors.get(level, self.colors['info'])
        
        self.console.print(
            f"[dim]{timestamp}[/] "
            f"[bold {color}][{level:>7}][/] "
            f"[white]{message}[/]"
        )
    
    def show_data_info(self, data_info: Dict):
        """Display dataset information in a table."""
        table = Table(
            title="📊 DATASET INFORMATION",
            title_style=f"bold {self.colors['neon_green']}",
            border_style=self.colors['electric_blue'],
            box=ROUNDED,
            show_header=True,
            header_style=f"bold {self.colors['cyber_yellow']}"
        )
        
        table.add_column("Property", style=self.colors['hot_pink'], width=20)
        table.add_column("Value", style="white", width=30)
        
        for key, value in data_info.items():
            table.add_row(str(key).replace('_', ' ').title(), str(value))
        
        self.console.print(table)
        self.console.print()
    
    def show_genome_table(self, genomes: List, top_n: int = 5):
        """Display top genomes in a table."""
        table = Table(
            title=f"🧬 TOP {top_n} EVOLVED ARCHITECTURES",
            title_style=f"bold {self.colors['neon_green']}",
            border_style=self.colors['deep_purple'],
            box=HEAVY,
            show_header=True,
            header_style=f"bold {self.colors['cyber_yellow']}"
        )
        
        table.add_column("Rank", style=self.colors['hot_pink'], width=6)
        table.add_column("Architecture", style=self.colors['electric_blue'], width=30)
        table.add_column("Activations", style=self.colors['neon_green'], width=25)
        table.add_column("Learning Rate", style="white", width=12)
        table.add_column("Fitness", style=self.colors['success'], width=10)
        
        sorted_genomes = sorted(genomes, key=lambda g: g.fitness, reverse=True)[:top_n]
        
        for i, genome in enumerate(sorted_genomes, 1):
            architecture = " → ".join([str(n) for n in genome.hidden_layers])
            activations = ", ".join(genome.activation_functions[:3])
            if len(genome.activation_functions) > 3:
                activations += "..."
            
            table.add_row(
                f"#{i}",
                architecture,
                activations,
                f"{genome.learning_rate:.6f}",
                f"{genome.fitness:.4f}"
            )
        
        self.console.print(table)
        self.console.print()
    
    def show_training_progress(self, epoch: int, total_epochs: int, 
                              train_loss: float, val_loss: float,
                              train_acc: float = None, val_acc: float = None):
        """Display training progress with metrics."""
        progress_pct = (epoch / total_epochs) * 100
        
        # Create progress bar
        bar_width = 40
        filled = int(bar_width * epoch / total_epochs)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        self.console.print(
            f"[{self.colors['electric_blue']}]Epoch [{epoch:>3}/{total_epochs}] "
            f"[{self.colors['neon_green']}]{bar}[/] "
            f"[{self.colors['cyber_yellow']}]{progress_pct:.1f}%[/]"
        )
        
        # Metrics
        metrics_text = f"  Loss: Train={train_loss:.4f}, Val={val_loss:.4f}"
        if train_acc is not None and val_acc is not None:
            metrics_text += f" | Acc: Train={train_acc:.4f}, Val={val_acc:.4f}"
        
        self.console.print(f"[white]{metrics_text}[/]")
    
    def show_evolution_progress(self, generation: int, total_generations: int,
                               best_fitness: float, avg_fitness: float,
                               diversity: float):
        """Display genetic algorithm evolution progress."""
        panel = Panel(
            f"[{self.colors['neon_green']}]🧬 Generation: {generation}/{total_generations}[/]\n"
            f"[{self.colors['electric_blue']}]Best Fitness: {best_fitness:.6f}[/]\n"
            f"[{self.colors['hot_pink']}]Avg Fitness:  {avg_fitness:.6f}[/]\n"
            f"[{self.colors['cyber_yellow']}]Diversity:   {diversity:.6f}[/]",
            title=f"[bold {self.colors['deep_purple']}]EVOLUTION STATUS[/]",
            border_style=self.colors['neon_green'],
            box=ROUNDED
        )
        self.console.print(panel)
    
    def show_architecture_summary(self, summary: str):
        """Display network architecture summary."""
        panel = Panel(
            summary,
            title="[bold]🏗️  NEURAL ARCHITECTURE[/]",
            title_align="left",
            border_style=self.colors['neon_green'],
            box=DOUBLE
        )
        self.console.print(panel)
    
    def show_math_explanation(self, equation_name: str, equation: str, description: str):
        """
        Display mathematical equations with explanations.
        
        Args:
            equation_name: Name of the equation
            equation: LaTeX-style equation (displayed as text)
            description: Plain English explanation
        """
        self.console.print()
        self.console.print(f"[bold {self.colors['cyber_yellow']}]📐 {equation_name}[/]")
        self.show_divider("─", 60, self.colors['deep_purple'])
        
        # Equation box
        eq_panel = Panel(
            f"[{self.colors['electric_blue']}]{equation}[/]",
            border_style=self.colors['hot_pink'],
            box=ROUNDED,
            expand=False
        )
        self.console.print(eq_panel)
        
        # Description
        self.console.print(f"[white]{description}[/]")
        self.console.print()
    
    def show_results_dashboard(self, results: Dict):
        """Display final results in a comprehensive dashboard."""
        self.clear_screen()
        self.show_header("FINAL RESULTS DASHBOARD", "Mission Complete: Neural Architecture Evolved")
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="metrics", size=10),
            Layout(name="architecture", size=15)
        )
        
        # Metrics panel
        metrics_table = Table(
            show_header=False,
            border_style=self.colors['neon_green'],
            box=ROUNDED
        )
        metrics_table.add_column("Metric", style=f"bold {self.colors['electric_blue']}")
        metrics_table.add_column("Value", style=f"bold {self.colors['hot_pink']}")
        
        if 'test_metrics' in results:
            for key, value in results['test_metrics'].items():
                metrics_table.add_row(key.upper(), f"{value:.6f}")
        
        metrics_panel = Panel(
            metrics_table,
            title="[bold]📈 TEST PERFORMANCE[/]",
            border_style=self.colors['success'],
            box=DOUBLE
        )
        
        # Architecture panel
        arch_text = ""
        if 'best_genome' in results:
            genome = results['best_genome']
            arch_text = (
                f"[{self.colors['neon_green']}]Architecture:[/] "
                f"{' → '.join([str(n) for n in genome.get('hidden_layers', [])])}\n"
                f"[{self.colors['electric_blue']}]Activations:[/] "
                f"{', '.join(genome.get('activation_functions', []))}\n"
                f"[{self.colors['hot_pink']}]Learning Rate:[/] "
                f"{genome.get('learning_rate', 0):.6f}\n"
                f"[{self.colors['cyber_yellow']}]Fitness:[/] "
                f"{genome.get('fitness', 0):.6f}"
            )
        
        arch_panel = Panel(
            arch_text,
            title="[bold]🧬 EVOLVED ARCHITECTURE[/]",
            border_style=self.colors['deep_purple'],
            box=DOUBLE
        )
        
        self.console.print(metrics_panel)
        self.console.print(arch_panel)
        self.console.print()
    
    def show_completion_banner(self):
        """Display mission complete banner."""
        banner = pyfiglet.figlet_format("SUCCESS", font="banner3")
        
        panel = Panel(
            f"[bold {self.colors['success']}]{banner}[/]\n"
            f"[{self.colors['electric_blue']}]Neural Evolution Successful[/]\n"
            f"[{self.colors['neon_green']}]All Systems Nominal[/]\n"
            f"[{self.colors['cyber_yellow']}]Output Files Generated[/]",
            border_style=self.colors['hot_pink'],
            box=DOUBLE,
            expand=False
        )
        
        self.console.print(Align.center(panel))
    
    def create_progress_bar(self, description: str, total: int):
        """
        Create a rich progress bar context manager.
        
        Args:
            description: Progress bar description
            total: Total steps
            
        Returns:
            Progress context manager
        """
        return Progress(
            SpinnerColumn(spinner_name="dots12", style=self.colors['electric_blue']),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                complete_style=self.colors['neon_green'],
                finished_style=self.colors['success']
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
    
    def pause(self, message: str = "Press Enter to continue..."):
        """Pause execution and wait for user input."""
        self.console.input(f"[{self.colors['cyber_yellow']}]{message}[/]")
    
    def confirm(self, message: str) -> bool:
        """
        Ask for user confirmation.
        
        Args:
            message: Confirmation message
            
        Returns:
            True if confirmed, False otherwise
        """
        response = self.console.input(
            f"[{self.colors['cyber_yellow']}]{message} (y/n): [/]"
        ).strip().lower()
        return response in ['y', 'yes']
    
    def show_main_menu(self) -> str:
        """
        Display interactive main menu and return user choice.
        
        Returns:
            User's menu selection
        """
        self.clear_screen()
        
        # Title
        title = pyfiglet.figlet_format("NEUROGEN", font="doom")
        self.console.print(title, style=f"bold {self.colors['neon_green']}")
        
        menu_panel = Panel(
            f"[bold {self.colors['electric_blue']}]█ NEUROGEN WineLab - CONTROL PANEL █[/]\n\n"
            f"[{self.colors['neon_green']}][1] 🧬 NEW RUN[/] - Evolve New Neural Architecture\n"
            f"[{self.colors['electric_blue']}][2] 💾 LOAD CORE[/] - Load Saved Model\n"
            f"[{self.colors['deep_purple']}][3] INFERENCE[/] - Test Model Predictions\n"
            f"[{self.colors['hot_pink']}][4] VIEW MODELS[/] - List Saved Models\n"
            f"[{self.colors['cyber_yellow']}][5] DEEP ANALYSIS[/] - Advanced Dataset & Model Analysis\n"
            f"[{self.colors['hot_pink']}][6] EXPLAIN MODEL[/] - What Does This Model Do?\n"
            f"[{self.colors['neon_green']}][7] INTERACTIVE TEST[/] - Test with Custom Wine Sample\n"
            f"[{self.colors['deep_purple']}][8] TOGGLE MODE[/] - Switch Classification/Regression\n"
            f"[{self.colors['error']}][9] EXIT[/] - Shutdown System\n\n"
            f"[dim]Select your operation...[/]",
            border_style=self.colors['neon_green'],
            box=HEAVY,
            expand=False,
            padding=(1, 2)
        )
        
        self.console.print(Align.center(menu_panel))
        self.console.print()
        
        choice = self.console.input(
            f"[bold {self.colors['cyber_yellow']}]>>> ENTER COMMAND: [/]"
        ).strip()
        
        return choice
    
    def show_model_selection(self, models: List[Dict]) -> Optional[int]:
        """
        Display list of saved models for selection.
        
        Args:
            models: List of model dictionaries
            
        Returns:
            Selected model index or None
        """
        if not models:
            self.log("No saved models found.", "warning")
            return None
        
        self.clear_screen()
        self.show_header("SAVED NEURAL CORES", "Select a model to load")
        
        table = Table(
            title="",
            box=DOUBLE,
            border_style=self.colors['electric_blue'],
            header_style=f"bold {self.colors['neon_green']}"
        )
        
        table.add_column("#", style=self.colors['cyber_yellow'], justify="center")
        table.add_column("Filename", style=self.colors['electric_blue'])
        table.add_column("Timestamp", style=self.colors['deep_purple'])
        table.add_column("Fitness", style=self.colors['neon_green'], justify="right")
        table.add_column("Accuracy", style=self.colors['hot_pink'], justify="right")
        
        for idx, model in enumerate(models, 1):
            table.add_row(
                str(idx),
                model['filename'][:40],
                model['timestamp'][:19] if len(model['timestamp']) > 19 else model['timestamp'],
                f"{model.get('fitness', 0):.4f}",
                f"{model.get('accuracy', 0):.4f}"
            )
        
        self.console.print(table)
        self.console.print()
        
        choice = self.console.input(
            f"[{self.colors['cyber_yellow']}]Select model number (or 'q' to quit): [/]"
        ).strip()
        
        if choice.lower() == 'q':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return idx
            else:
                self.log("Invalid selection", "error")
                return None
        except ValueError:
            self.log("Invalid input", "error")
            return None
    
    def show_inference_scanning(self):
        """Display scanning animation for inference mode."""
        self.clear_screen()
        
        from rich.text import Text
        scanner_content = Text()
        scanner_content.append("\n🔮 INFERENCE MODE ACTIVATED 🔮\n", style=f"bold {self.colors['neon_green']}")
        scanner_content.append("\n")
        scanner_content.append("  Scanning Neural Network State...\n", style=self.colors['cyber_yellow'])
        scanner_content.append("  Loading Synaptic Weights...\n", style=self.colors['electric_blue'])
        scanner_content.append("  Calibrating Activation Functions...\n", style=self.colors['hot_pink'])
        scanner_content.append("\n")
        scanner_content.append("       [████████████████]\n", style=f"bold {self.colors['success']}")
        scanner_content.append("\n")
        
        panel = Panel(
            Align.center(scanner_content),
            border_style=self.colors['deep_purple'],
            box=HEAVY,
            padding=(1, 2)
        )
        
        self.console.print(Align.center(panel))
        time.sleep(2)
    
    def show_inference_results(self, results: List[Dict]):
        """
        Display inference results in a beautiful table.
        
        Args:
            results: List of inference result dictionaries
        """
        self.clear_screen()
        self.show_header("🔮 INFERENCE RESULTS", "Neural Network Predictions")
        
        table = Table(
            box=DOUBLE,
            border_style=self.colors['deep_purple'],
            header_style=f"bold {self.colors['neon_green']}",
            show_lines=True
        )
        
        table.add_column("Sample", style=self.colors['cyber_yellow'], justify="center")
        table.add_column("Input Features", style=self.colors['electric_blue'])
        table.add_column("True Label", style=self.colors['neon_green'], justify="center")
        table.add_column("Prediction", style=self.colors['hot_pink'], justify="center")
        table.add_column("Confidence", style=self.colors['deep_purple'], justify="right")
        table.add_column("Status", justify="center")
        
        for idx, result in enumerate(results, 1):
            # Format input features (show first 3 values)
            features = result.get('features', [])
            feature_str = ", ".join([f"{f:.2f}" for f in features[:3]]) + "..."
            
            true_label = result.get('true_label', 'N/A')
            prediction = result.get('prediction', 'N/A')
            confidence = result.get('confidence', 0.0)
            
            # Determine status with color
            if true_label == prediction:
                status = f"[bold {self.colors['success']}]✓ MATCH[/]"
            else:
                status = f"[bold {self.colors['error']}]✗ ERROR[/]"
            
            table.add_row(
                f"#{idx}",
                feature_str,
                str(true_label),
                str(prediction),
                f"{confidence:.1%}",
                status
            )
        
        self.console.print(table)
        self.console.print()
        
        # Calculate accuracy
        matches = sum(1 for r in results if r.get('true_label') == r.get('prediction'))
        accuracy = matches / len(results) if results else 0
        
        acc_panel = Panel(
            f"[bold {self.colors['neon_green']}]Inference Accuracy: {accuracy:.1%}[/]\n"
            f"[{self.colors['electric_blue']}]Correct Predictions: {matches}/{len(results)}[/]",
            border_style=self.colors['hot_pink'],
            box=ROUNDED
        )
        
        self.console.print(acc_panel)
    
    def show_loading_animation(self, message: str = "Loading", duration: float = 2.0):
        """Display a loading animation."""
        with Progress(
            SpinnerColumn(spinner_name="dots", style=self.colors['electric_blue']),
            TextColumn(f"[{self.colors['neon_green']}]{message}...[/]"),
            console=self.console
        ) as progress:
            task = progress.add_task("", total=100)
            for i in range(100):
                time.sleep(duration / 100)
                progress.update(task, advance=1)
    
    def get_wine_features_interactive(self) -> Optional[Dict[str, float]]:
        """
        Interactive input for wine features.
        
        Returns:
            Dictionary with feature values or None if cancelled
        """
        self.clear_screen()
        self.show_header("INTERACTIVE WINE TEST", "Enter wine characteristics for prediction")
        
        feature_names = [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]
        
        feature_ranges = {
            "fixed_acidity": (4.0, 16.0, 7.0),
            "volatile_acidity": (0.1, 1.6, 0.5),
            "citric_acid": (0.0, 1.0, 0.3),
            "residual_sugar": (0.5, 15.0, 2.5),
            "chlorides": (0.01, 0.6, 0.08),
            "free_sulfur_dioxide": (1.0, 72.0, 15.0),
            "total_sulfur_dioxide": (6.0, 290.0, 46.0),
            "density": (0.99, 1.01, 0.996),
            "pH": (2.7, 4.0, 3.3),
            "sulphates": (0.3, 2.0, 0.65),
            "alcohol": (8.0, 15.0, 10.5)
        }
        
        features = {}
        
        self.console.print(
            f"[{self.colors['cyber_yellow']}]Enter values for each feature (or press Enter for default)[/]\n"
        )
        
        for feature in feature_names:
            min_val, max_val, default = feature_ranges[feature]
            
            while True:
                prompt = (
                    f"[{self.colors['neon_green']}]{feature}[/] "
                    f"[dim]({min_val:.2f} - {max_val:.2f}, default: {default:.2f})[/]: "
                )
                
                user_input = self.console.input(prompt).strip()
                
                if user_input.lower() == 'q':
                    return None
                
                if user_input == "":
                    features[feature] = default
                    break
                
                try:
                    value = float(user_input)
                    if min_val <= value <= max_val:
                        features[feature] = value
                        break
                    else:
                        self.log(f"Value out of range ({min_val:.2f} - {max_val:.2f})", "WARNING")
                except ValueError:
                    self.log("Invalid number format", "ERROR")
        
        return features
    
    def show_prediction_result(self, features: Dict[str, float], prediction: any, 
                              probabilities: Optional[list] = None, task: str = "classification"):
        """
        Display prediction result for interactive test.
        
        Args:
            features: Input features
            prediction: Model prediction
            probabilities: Class probabilities (for classification)
            task: Task type (classification or regression)
        """
        self.clear_screen()
        self.show_header("PREDICTION RESULT", "Wine Quality Prediction")
        
        # Features table
        feat_table = Table(
            title="Input Features",
            box=ROUNDED,
            border_style=self.colors['electric_blue']
        )
        feat_table.add_column("Feature", style=self.colors['neon_green'])
        feat_table.add_column("Value", style=self.colors['cyber_yellow'], justify="right")
        
        for feat, val in features.items():
            feat_table.add_row(feat, f"{val:.3f}")
        
        self.console.print(feat_table)
        self.console.print()
        
        # Prediction result
        if task == "classification":
            class_names = ["LOW Quality (3-5)", "MEDIUM Quality (5-7)", "HIGH Quality (7-9)"]
            
            result_text = f"[bold {self.colors['neon_green']}]Predicted Class: {prediction}[/]\n"
            result_text += f"[{self.colors['electric_blue']}]{class_names[prediction]}[/]\n\n"
            
            if probabilities is not None:
                result_text += f"[bold {self.colors['cyber_yellow']}]Class Probabilities:[/]\n"
                for i, prob in enumerate(probabilities):
                    bar_length = int(prob * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    result_text += f"  Class {i}: {bar} {prob:.1%}\n"
        else:
            result_text = f"[bold {self.colors['neon_green']}]Predicted Quality Score: {prediction:.2f}[/]\n"
            result_text += f"[{self.colors['electric_blue']}]Wine quality on scale 0-10[/]"
        
        result_panel = Panel(
            result_text,
            title="[bold]🎯 PREDICTION[/]",
            border_style=self.colors['hot_pink'],
            box=DOUBLE
        )
        
        self.console.print(result_panel)
    
    def show_mode_toggle(self, current_mode: str) -> bool:
        """
        Display mode toggle confirmation.
        
        Args:
            current_mode: Current task mode
            
        Returns:
            True if user confirms toggle, False otherwise
        """
        self.clear_screen()
        self.show_header("MODE TOGGLE", "Switch between Classification and Regression")
        
        new_mode = "regression" if current_mode == "classification" else "classification"
        
        info_text = f"[bold {self.colors['cyber_yellow']}]Current Mode:[/] "
        info_text += f"[{self.colors['neon_green']}]{current_mode.upper()}[/]\n\n"
        info_text += f"[bold {self.colors['cyber_yellow']}]New Mode:[/] "
        info_text += f"[{self.colors['hot_pink']}]{new_mode.upper()}[/]\n\n"
        
        if new_mode == "regression":
            info_text += f"[{self.colors['electric_blue']}]Regression predicts continuous quality scores (0-10)[/]\n"
        else:
            info_text += f"[{self.colors['electric_blue']}]Classification predicts quality classes (LOW/MEDIUM/HIGH)[/]\n"
        
        info_text += f"\n[dim]Note: You'll need to run NEW RUN to train a model in the new mode[/]"
        
        panel = Panel(
            info_text,
            border_style=self.colors['deep_purple'],
            box=DOUBLE
        )
        
        self.console.print(panel)
        self.console.print()
        
        return self.confirm(f"Switch to {new_mode.upper()} mode?")
