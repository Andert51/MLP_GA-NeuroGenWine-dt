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
        banner = pyfiglet.figlet_format("VINOGEN", font="slant")
        subtitle = pyfiglet.figlet_format("CYBERCORE", font="digital")
        
        self.console.print(banner, style=f"bold {self.colors['neon_green']}")
        self.console.print(subtitle, style=f"bold {self.colors['electric_blue']}")
        self.console.print()
        
        # Boot messages
        boot_messages = [
            ("[SYSTEM]", "Initializing Neural Core...", self.colors['info']),
            ("[QUANTUM]", "Loading Genetic Algorithm Engine...", self.colors['electric_blue']),
            ("[MATRIX]", "Establishing Data Pipeline...", self.colors['neon_green']),
            ("[CYBER]", "Activating Visualization Engine...", self.colors['deep_purple']),
            ("[NEURAL]", "Calibrating Synaptic Weights...", self.colors['hot_pink']),
            ("[AI]", "Deploying Evolution Protocol...", self.colors['cyber_yellow']),
            ("[READY]", "VinoGen-CyberCore Online!", self.colors['success'])
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
        banner = pyfiglet.figlet_format("COMPLETE", font="banner3")
        
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
