"""
VinoGen-CyberCore: Model Explainer
Visualizations to explain what the model does and how it works.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import torch
from pathlib import Path


class ModelExplainer:
    """
    Creates educational visualizations to explain:
    1. What the model does (classification vs regression)
    2. How inputs become outputs
    3. What each prediction means
    4. Feature importance and relationships
    """
    
    def __init__(self, output_dir: Path = None):
        """Initialize the explainer."""
        self.output_dir = output_dir or Path("output/explanations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set cyberpunk theme
        plt.style.use('dark_background')
        self.colors = {
            'bg': '#1a1a2e',
            'primary': '#00ff9f',
            'secondary': '#00d9ff',
            'accent': '#ff006e',
            'warning': '#ffbe0b',
            'text': '#e0e0e0'
        }
    
    def explain_model_task(
        self, 
        task: str = "classification",
        n_classes: int = 3,
        feature_names: List[str] = None,
        save_path: str = None
    ) -> str:
        """
        Create a comprehensive visual explanation of what the model does.
        
        Shows:
        - Input features (11 wine chemical properties)
        - Model process (neural network)
        - Output meaning (classes or continuous values)
        - Task comparison (classification vs regression)
        
        Args:
            task: "classification" or "regression"
            n_classes: Number of output classes (for classification)
            feature_names: List of input feature names
            save_path: Where to save the figure
            
        Returns:
            Path to saved figure
        """
        if feature_names is None:
            feature_names = [
                'Acidez Fija', 'Acidez Volátil', 'Ácido Cítrico',
                'Azúcar Residual', 'Cloruros', 'SO₂ Libre',
                'SO₂ Total', 'Densidad', 'pH', 'Sulfatos', 'Alcohol'
            ]
        
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor(self.colors['bg'])
        
        # Title
        fig.suptitle(
            '🔬 ¿QUÉ HACE ESTE MODELO DE RED NEURONAL?',
            fontsize=28, fontweight='bold', color=self.colors['primary'], y=0.98
        )
        
        # Create 3x2 grid
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25,
                             left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        # ============================================================
        # PANEL 1: INPUT FEATURES (Top Left)
        # ============================================================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        title_text = ax1.text(
            0.5, 0.95, '📊 ENTRADA: 11 Propiedades Químicas del Vino',
            ha='center', fontsize=16, fontweight='bold',
            color=self.colors['secondary'], transform=ax1.transAxes
        )
        
        # Draw input features as colored boxes
        y_pos = 0.85
        for i, feature in enumerate(feature_names):
            # Box
            rect = plt.Rectangle((0.1, y_pos - 0.04), 0.8, 0.06,
                                facecolor=self.colors['primary'], alpha=0.2,
                                edgecolor=self.colors['primary'], linewidth=2,
                                transform=ax1.transAxes)
            ax1.add_patch(rect)
            
            # Feature name
            ax1.text(0.15, y_pos, feature, fontsize=11, color=self.colors['text'],
                    transform=ax1.transAxes, va='center')
            
            # Example value
            example_val = ['7.4', '0.70', '0.0', '1.9', '0.076', '11', 
                          '34', '0.998', '3.51', '0.56', '9.4'][i]
            ax1.text(0.85, y_pos, f'ej: {example_val}', fontsize=10,
                    color=self.colors['warning'], transform=ax1.transAxes,
                    va='center', ha='right')
            
            y_pos -= 0.075
        
        # Arrow to next panel
        ax1.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=3, color=self.colors['primary']))
        
        # ============================================================
        # PANEL 2: NEURAL NETWORK PROCESS (Top Right)
        # ============================================================
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        ax2.text(
            0.5, 0.95, '🧠 PROCESO: Red Neuronal MLP',
            ha='center', fontsize=16, fontweight='bold',
            color=self.colors['secondary'], transform=ax2.transAxes
        )
        
        # Draw simplified neural network
        layer_sizes = [11, 8, 6, n_classes if task == "classification" else 1]
        layer_names = ['Input\n(11)', 'Capa 1\n(~64)', 'Capa 2\n(~32)', 
                      f'Output\n({n_classes} clases)' if task == "classification" else 'Output\n(1 valor)']
        
        x_positions = np.linspace(0.15, 0.85, len(layer_sizes))
        y_center = 0.5
        
        for i, (size, name, x_pos) in enumerate(zip(layer_sizes, layer_names, x_positions)):
            # Draw neurons
            y_positions = np.linspace(0.2, 0.8, min(size, 5))
            
            for y_pos in y_positions:
                circle = plt.Circle((x_pos, y_pos), 0.03,
                                   facecolor=self.colors['bg'],
                                   edgecolor=self.colors['primary'],
                                   linewidth=2, transform=ax2.transAxes,
                                   zorder=3)
                ax2.add_patch(circle)
            
            # Layer label
            ax2.text(x_pos, 0.05, name, ha='center', fontsize=10,
                    color=self.colors['text'], transform=ax2.transAxes)
            
            # Connections to next layer
            if i < len(layer_sizes) - 1:
                next_x = x_positions[i + 1]
                for y1 in y_positions:
                    for y2 in np.linspace(0.2, 0.8, min(layer_sizes[i+1], 5)):
                        ax2.plot([x_pos, next_x], [y1, y2],
                               color=self.colors['primary'], alpha=0.1,
                               linewidth=0.5, transform=ax2.transAxes, zorder=1)
        
        # Process description
        process_text = (
            "• Forward propagation\n"
            "• Activation functions (ReLU, Tanh)\n"
            "• Pesos optimizados por algoritmo genético\n"
            "• Backpropagation para ajuste fino"
        )
        ax2.text(0.5, 0.92, process_text, ha='center', fontsize=9,
                color=self.colors['warning'], transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor=self.colors['bg'],
                         edgecolor=self.colors['warning'], alpha=0.3))
        
        # Arrow down
        ax2.annotate('', xy=(0.5, 0.0), xytext=(0.5, 0.1),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=3, color=self.colors['primary']))
        
        # ============================================================
        # PANEL 3: OUTPUT EXPLANATION (Middle Left)
        # ============================================================
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        if task == "classification":
            ax3.text(
                0.5, 0.95, f'🎯 SALIDA: {n_classes} Clases de Calidad',
                ha='center', fontsize=16, fontweight='bold',
                color=self.colors['secondary'], transform=ax3.transAxes
            )
            
            # Show classes
            class_names = ['BAJA', 'MEDIA', 'ALTA'] if n_classes == 3 else \
                         [f'Clase {i}' for i in range(n_classes)]
            class_colors = [self.colors['accent'], self.colors['warning'], self.colors['primary']]
            
            y_pos = 0.75
            for i, (name, color) in enumerate(zip(class_names, class_colors[:n_classes])):
                # Class box
                rect = plt.Rectangle((0.15, y_pos - 0.05), 0.7, 0.12,
                                   facecolor=color, alpha=0.3,
                                   edgecolor=color, linewidth=3,
                                   transform=ax3.transAxes)
                ax3.add_patch(rect)
                
                # Class label
                ax3.text(0.2, y_pos, f'Clase {i}: {name}',
                        fontsize=14, fontweight='bold', color=color,
                        transform=ax3.transAxes, va='center')
                
                # Quality range
                quality_ranges = ['3-5', '5-7', '7-9']
                if i < len(quality_ranges):
                    ax3.text(0.8, y_pos, f'Calidad: {quality_ranges[i]}',
                            fontsize=11, color=self.colors['text'],
                            transform=ax3.transAxes, va='center', ha='right')
                
                y_pos -= 0.2
            
            # Explanation
            explanation = (
                "El modelo NO predice un número exacto\n"
                "sino que CLASIFICA el vino en una de\n"
                f"las {n_classes} categorías de calidad."
            )
            ax3.text(0.5, 0.05, explanation, ha='center', fontsize=11,
                    color=self.colors['warning'], transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor=self.colors['bg'],
                             edgecolor=self.colors['warning'], alpha=0.5))
        
        else:  # regression
            ax3.text(
                0.5, 0.95, '📈 SALIDA: Valor Continuo de Calidad',
                ha='center', fontsize=16, fontweight='bold',
                color=self.colors['secondary'], transform=ax3.transAxes
            )
            
            # Draw continuous scale
            scale_y = 0.6
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax3.imshow(gradient, aspect='auto', cmap='RdYlGn',
                      extent=[0.1, 0.9, scale_y - 0.05, scale_y + 0.05],
                      transform=ax3.transAxes)
            
            # Labels
            ax3.text(0.1, scale_y + 0.1, '0', ha='center', fontsize=12,
                    color=self.colors['text'], transform=ax3.transAxes)
            ax3.text(0.9, scale_y + 0.1, '10', ha='center', fontsize=12,
                    color=self.colors['text'], transform=ax3.transAxes)
            ax3.text(0.5, scale_y + 0.1, 'Calidad del Vino', ha='center',
                    fontsize=13, fontweight='bold', color=self.colors['primary'],
                    transform=ax3.transAxes)
            
            # Example predictions
            examples = [
                ('Ej: 5.3', 0.53, self.colors['accent']),
                ('Ej: 7.8', 0.78, self.colors['primary']),
                ('Ej: 6.1', 0.61, self.colors['warning'])
            ]
            
            for text, pos, color in examples:
                x_pos = 0.1 + 0.8 * (pos / 10)
                ax3.plot([x_pos, x_pos], [scale_y - 0.05, scale_y - 0.15],
                        color=color, linewidth=2, transform=ax3.transAxes)
                ax3.text(x_pos, scale_y - 0.2, text, ha='center',
                        fontsize=11, color=color, transform=ax3.transAxes)
            
            # Explanation
            explanation = (
                "El modelo predice un NÚMERO EXACTO\n"
                "entre 0 y 10, representando la calidad\n"
                "continua del vino (ej: 6.3, 7.8, etc.)"
            )
            ax3.text(0.5, 0.15, explanation, ha='center', fontsize=11,
                    color=self.colors['warning'], transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor=self.colors['bg'],
                             edgecolor=self.colors['warning'], alpha=0.5))
        
        # ============================================================
        # PANEL 4: COMPARISON (Middle Right)
        # ============================================================
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        ax4.text(
            0.5, 0.95, '⚖️ CLASIFICACIÓN vs REGRESIÓN',
            ha='center', fontsize=16, fontweight='bold',
            color=self.colors['secondary'], transform=ax4.transAxes
        )
        
        # Comparison table
        comparison_data = [
            ('CLASIFICACIÓN', 'REGRESIÓN'),
            ('─' * 20, '─' * 20),
            ('Salida: Categorías', 'Salida: Números continuos'),
            ('ej: "BAJA", "MEDIA", "ALTA"', 'ej: 5.3, 6.7, 7.9'),
            ('', ''),
            ('✓ Agrupa vinos similares', '✓ Predice calidad exacta'),
            ('✓ Más simple de interpretar', '✓ Más información detallada'),
            ('✓ Robusto a pequeños errores', '✓ Útil para rankings precisos'),
            ('', ''),
            ('Métrica: Accuracy', 'Métrica: MAE, RMSE, R²'),
            ('ej: 87% correctas', 'ej: Error ±0.5 puntos'),
        ]
        
        y_pos = 0.85
        for i, (left, right) in enumerate(comparison_data):
            # Left (Classification)
            color_left = self.colors['primary'] if i == 0 else self.colors['text']
            ax4.text(0.25, y_pos, left, ha='center', fontsize=10,
                    color=color_left, transform=ax4.transAxes,
                    fontweight='bold' if i == 0 else 'normal')
            
            # Right (Regression)
            color_right = self.colors['warning'] if i == 0 else self.colors['text']
            ax4.text(0.75, y_pos, right, ha='center', fontsize=10,
                    color=color_right, transform=ax4.transAxes,
                    fontweight='bold' if i == 0 else 'normal')
            
            y_pos -= 0.075
        
        # Current mode indicator
        current_mode = "CLASIFICACIÓN" if task == "classification" else "REGRESIÓN"
        current_color = self.colors['primary'] if task == "classification" else self.colors['warning']
        
        ax4.text(0.5, 0.05, f'⚙️ MODO ACTUAL: {current_mode}',
                ha='center', fontsize=13, fontweight='bold',
                color=current_color, transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor=current_color,
                         edgecolor=current_color, alpha=0.3, linewidth=3))
        
        # ============================================================
        # PANEL 5: EXAMPLE PREDICTION (Bottom Left)
        # ============================================================
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        
        ax5.text(
            0.5, 0.95, '🍷 EJEMPLO DE PREDICCIÓN',
            ha='center', fontsize=16, fontweight='bold',
            color=self.colors['secondary'], transform=ax5.transAxes
        )
        
        # Example wine
        example_wine = {
            'Acidez Fija': 7.4,
            'Acidez Volátil': 0.70,
            'Ácido Cítrico': 0.0,
            'Azúcar': 1.9,
            'Cloruros': 0.076,
            'SO₂ Libre': 11,
            'SO₂ Total': 34,
            'Densidad': 0.998,
            'pH': 3.51,
            'Sulfatos': 0.56,
            'Alcohol': 9.4
        }
        
        # Input box
        input_text = "Vino de ejemplo (propiedades químicas):\n"
        for feature, value in list(example_wine.items())[:6]:
            input_text += f"  • {feature}: {value}\n"
        input_text += "  • ..."
        
        ax5.text(0.05, 0.7, input_text, fontsize=10,
                color=self.colors['text'], transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor=self.colors['bg'],
                         edgecolor=self.colors['secondary'], alpha=0.5),
                family='monospace')
        
        # Arrow
        ax5.annotate('', xy=(0.5, 0.5), xytext=(0.3, 0.6),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=3, color=self.colors['primary']))
        
        # Output box
        if task == "classification":
            output_text = (
                "🎯 PREDICCIÓN:\n\n"
                "  Clase 1: MEDIA (67%)\n"
                "  Clase 0: BAJA (28%)\n"
                "  Clase 2: ALTA (5%)\n\n"
                "✓ Vino de calidad MEDIA"
            )
            output_color = self.colors['warning']
        else:
            output_text = (
                "📈 PREDICCIÓN:\n\n"
                "  Calidad: 5.8/10\n\n"
                "  (Rango: [5.2, 6.4])\n\n"
                "✓ Vino aceptable"
            )
            output_color = self.colors['warning']
        
        ax5.text(0.55, 0.5, output_text, fontsize=11,
                color=output_color, transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor=output_color,
                         edgecolor=output_color, alpha=0.2, linewidth=2),
                family='monospace', va='center')
        
        # ============================================================
        # PANEL 6: HOW TO USE (Bottom Right)
        # ============================================================
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        ax6.text(
            0.5, 0.95, '📖 CÓMO USAR EL MODELO',
            ha='center', fontsize=16, fontweight='bold',
            color=self.colors['secondary'], transform=ax6.transAxes
        )
        
        # Instructions
        instructions = [
            ('1️⃣', 'ENTRENAR', 'Opción [1] NEW RUN - Evoluciona arquitectura'),
            ('2️⃣', 'EVALUAR', 'El sistema prueba automáticamente en test set'),
            ('3️⃣', 'PREDECIR', 'Opción [3] INFERENCE - Prueba con nuevos vinos'),
            ('4️⃣', 'ANALIZAR', 'Opción [5] DEEP ANALYSIS - Visualizaciones'),
        ]
        
        y_pos = 0.8
        for emoji, title, desc in instructions:
            ax6.text(0.05, y_pos, emoji, fontsize=16,
                    transform=ax6.transAxes, va='center')
            ax6.text(0.15, y_pos, title, fontsize=12, fontweight='bold',
                    color=self.colors['primary'], transform=ax6.transAxes, va='center')
            ax6.text(0.15, y_pos - 0.05, desc, fontsize=9,
                    color=self.colors['text'], transform=ax6.transAxes, va='top')
            y_pos -= 0.18
        
        # Change task info
        change_text = (
            "💡 Para cambiar de CLASIFICACIÓN a REGRESIÓN:\n"
            "   Editar: src/utils/config.py\n"
            "   Cambiar: TASK = \"regression\""
        )
        ax6.text(0.5, 0.08, change_text, ha='center', fontsize=9,
                color=self.colors['warning'], transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor=self.colors['bg'],
                         edgecolor=self.colors['warning'], alpha=0.3),
                family='monospace')
        
        # Save
        if save_path is None:
            save_path = self.output_dir / "model_explanation.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor=self.colors['bg'])
        plt.close()
        
        return str(save_path)
    
    def explain_prediction_example(
        self,
        model,
        X_sample: np.ndarray,
        y_true: int or float,
        feature_names: List[str],
        task: str = "classification",
        class_names: List[str] = None,
        save_path: str = None
    ) -> str:
        """
        Show a detailed breakdown of a single prediction.
        
        Args:
            model: Trained model
            X_sample: Single input sample (11 features)
            y_true: True label/value
            feature_names: Names of input features
            task: "classification" or "regression"
            class_names: Names for classes (if classification)
            save_path: Where to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(self.colors['bg'])
        fig.suptitle(
            '🔍 DESGLOSE DETALLADO DE UNA PREDICCIÓN',
            fontsize=20, fontweight='bold', color=self.colors['primary']
        )
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sample.reshape(1, -1))
            output = model(X_tensor)
            
            if task == "classification":
                probas = torch.softmax(output, dim=1).numpy()[0]
                pred_class = np.argmax(probas)
            else:
                pred_value = output.numpy()[0, 0]
        
        # ============================================================
        # Panel 1: Input Features
        # ============================================================
        ax = axes[0, 0]
        ax.set_facecolor(self.colors['bg'])
        ax.set_title('📊 Valores de Entrada (11 Features)',
                    fontsize=14, fontweight='bold', color=self.colors['secondary'])
        
        y_pos = np.arange(len(feature_names))
        colors_bar = [self.colors['primary'] if x > 0.5 else self.colors['accent']
                     for x in X_sample]
        
        ax.barh(y_pos, X_sample, color=colors_bar, alpha=0.7,
               edgecolor='white', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, color=self.colors['text'])
        ax.set_xlabel('Valor Normalizado', color=self.colors['text'])
        ax.grid(axis='x', alpha=0.3, color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['text'])
        ax.spines['left'].set_color(self.colors['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # ============================================================
        # Panel 2: Model Output
        # ============================================================
        ax = axes[0, 1]
        ax.set_facecolor(self.colors['bg'])
        
        if task == "classification":
            ax.set_title('🎯 Salida del Modelo (Probabilidades)',
                        fontsize=14, fontweight='bold', color=self.colors['secondary'])
            
            if class_names is None:
                class_names = [f'Clase {i}' for i in range(len(probas))]
            
            colors_pred = [self.colors['primary'] if i == pred_class else self.colors['warning']
                          for i in range(len(probas))]
            
            bars = ax.bar(range(len(probas)), probas, color=colors_pred,
                         alpha=0.7, edgecolor='white', linewidth=2)
            ax.set_xticks(range(len(probas)))
            ax.set_xticklabels(class_names, color=self.colors['text'])
            ax.set_ylabel('Probabilidad', color=self.colors['text'])
            ax.set_ylim([0, 1])
            
            # Add percentage labels
            for i, (bar, prob) in enumerate(zip(bars, probas)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{prob*100:.1f}%', ha='center', va='bottom',
                       color=self.colors['primary' if i == pred_class else 'text'],
                       fontsize=11, fontweight='bold')
            
            # Winner annotation
            ax.annotate('GANADOR ✓', xy=(pred_class, probas[pred_class]),
                       xytext=(pred_class, probas[pred_class] + 0.15),
                       ha='center', fontsize=12, fontweight='bold',
                       color=self.colors['primary'],
                       arrowprops=dict(arrowstyle='->', lw=2,
                                     color=self.colors['primary']))
        
        else:  # regression
            ax.set_title('📈 Salida del Modelo (Valor Predicho)',
                        fontsize=14, fontweight='bold', color=self.colors['secondary'])
            
            # Draw scale
            ax.set_xlim([0, 10])
            ax.set_ylim([0, 1])
            ax.axhline(0.5, color=self.colors['text'], alpha=0.3, linewidth=1)
            
            # Predicted value
            ax.scatter([pred_value], [0.5], s=500, c=[self.colors['primary']],
                      edgecolors='white', linewidths=3, zorder=3)
            ax.text(pred_value, 0.65, f'Predicción:\n{pred_value:.2f}',
                   ha='center', fontsize=12, fontweight='bold',
                   color=self.colors['primary'])
            
            # True value
            ax.scatter([y_true], [0.5], s=300, c=[self.colors['warning']],
                      edgecolors='white', linewidths=2, marker='s', zorder=3)
            ax.text(y_true, 0.35, f'Real:\n{y_true:.2f}',
                   ha='center', fontsize=11,
                   color=self.colors['warning'])
            
            # Error
            error = abs(pred_value - y_true)
            ax.plot([pred_value, y_true], [0.5, 0.5],
                   color=self.colors['accent'], linewidth=3, linestyle='--')
            ax.text((pred_value + y_true) / 2, 0.55, f'Error: {error:.2f}',
                   ha='center', fontsize=10, color=self.colors['accent'])
            
            ax.set_xlabel('Calidad del Vino (0-10)', color=self.colors['text'])
            ax.set_yticks([])
        
        ax.grid(axis='x', alpha=0.3, color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['text'])
        ax.spines['left'].set_color(self.colors['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # ============================================================
        # Panel 3: Result Summary
        # ============================================================
        ax = axes[1, 0]
        ax.axis('off')
        
        if task == "classification":
            # Classification summary
            is_correct = (pred_class == y_true)
            result_text = "✅ CORRECTO" if is_correct else "❌ ERROR"
            result_color = self.colors['primary'] if is_correct else self.colors['accent']
            
            summary = f"""
{'='*40}
RESUMEN DE LA PREDICCIÓN
{'='*40}

Entrada: {len(feature_names)} características químicas
Modelo: Red Neuronal (Clasificación)

RESULTADO:
  Predicción: {class_names[pred_class]} (Clase {pred_class})
  Realidad:   {class_names[y_true]} (Clase {y_true})
  Confianza:  {probas[pred_class]*100:.1f}%
  
  Estado: {result_text}

DISTRIBUCIÓN DE PROBABILIDADES:
"""
            for i, (name, prob) in enumerate(zip(class_names, probas)):
                bar = '█' * int(prob * 20)
                summary += f"  {name:8s}: {bar:20s} {prob*100:5.1f}%\n"
            
            ax.text(0.05, 0.95, summary, fontsize=10, family='monospace',
                   color=self.colors['text'], transform=ax.transAxes,
                   va='top',
                   bbox=dict(boxstyle='round', facecolor=self.colors['bg'],
                            edgecolor=result_color, linewidth=3, alpha=0.8))
        
        else:  # regression
            error = abs(pred_value - y_true)
            rel_error = (error / max(abs(y_true), 0.01)) * 100
            
            summary = f"""
{'='*40}
RESUMEN DE LA PREDICCIÓN
{'='*40}

Entrada: {len(feature_names)} características químicas
Modelo: Red Neuronal (Regresión)

RESULTADO:
  Predicción: {pred_value:.2f}
  Realidad:   {y_true:.2f}
  
  Error Absoluto: {error:.2f}
  Error Relativo: {rel_error:.1f}%

INTERPRETACIÓN:
  {'Excelente' if error < 0.3 else 'Bueno' if error < 0.6 else 'Aceptable' if error < 1.0 else 'Mejorable'}
  (Error {'bajo' if error < 0.5 else 'moderado' if error < 1.0 else 'alto'})
"""
            
            result_color = self.colors['primary'] if error < 0.5 else \
                          self.colors['warning'] if error < 1.0 else self.colors['accent']
            
            ax.text(0.05, 0.95, summary, fontsize=10, family='monospace',
                   color=self.colors['text'], transform=ax.transAxes,
                   va='top',
                   bbox=dict(boxstyle='round', facecolor=self.colors['bg'],
                            edgecolor=result_color, linewidth=3, alpha=0.8))
        
        # ============================================================
        # Panel 4: Feature Importance (simplified)
        # ============================================================
        ax = axes[1, 1]
        ax.set_facecolor(self.colors['bg'])
        ax.set_title('🔥 Contribución de Features (Simplificado)',
                    fontsize=14, fontweight='bold', color=self.colors['secondary'])
        
        # Simple importance: absolute value of input * random weight proxy
        importance = np.abs(X_sample) * np.random.uniform(0.8, 1.2, len(X_sample))
        importance = importance / importance.sum()
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:8]  # Top 8
        
        colors_imp = [self.colors['primary'] if i in sorted_idx[:3] else self.colors['secondary']
                     for i in sorted_idx]
        
        ax.barh(range(len(sorted_idx)), importance[sorted_idx],
               color=colors_imp, alpha=0.7, edgecolor='white', linewidth=1.5)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx],
                          color=self.colors['text'])
        ax.set_xlabel('Importancia Relativa', color=self.colors['text'])
        ax.grid(axis='x', alpha=0.3, color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.spines['bottom'].set_color(self.colors['text'])
        ax.spines['left'].set_color(self.colors['text'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save
        if save_path is None:
            save_path = self.output_dir / "prediction_example.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor=self.colors['bg'])
        plt.close()
        
        return str(save_path)
