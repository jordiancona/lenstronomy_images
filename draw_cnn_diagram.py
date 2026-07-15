import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_net_diagram():
    # Ajustamos dimensiones para que todo quepa bien
    fig, ax = plt.subplots(figsize=(11, 22))
    ax.set_xlim(0, 14) 
    ax.set_ylim(-2, 24)
    ax.axis('off')
    
    # --- Configuración de Estilo ---
    box_width = 4.5      
    box_height = 1.
    x_center = 5
    start_y = 23
    gap = 1.5
    font_size_main = 14  
    font_size_dim = 14   
    
    # Colores
    colors = {
        'input': '#D3D3D3',   # Gris
        'conv': '#ADD8E6',    # Azul claro
        'pool': '#FFC0CB',    # Rosa
        'dense': '#90EE90',   # Verde claro
        'dropout': '#FFFFE0', # Amarillo claro
        'flatten': '#D3D3D3'  # Gris
    }

    # Clases en una sola línea
    classes_text = [r'$\theta_E$', r'$f$', r'$\epsilon_x$', r'$\epsilon_y$']
    classes_line = ",  ".join(classes_text) # Unión horizontal

    # --- Definición de Capas ---
    layers_data = [
        {'type': 'input', 'name': 'Input Image', 'desc': 'Grayscale', 'out': '(100, 100, 1)'},
        {'type': 'conv', 'name': 'Conv2D', 'desc': '128 filters, 7x7, /2\n+ BatchNorm + ReLU', 'out': '(47, 47, 128)'},
        {'type': 'pool', 'name': 'MaxPool2D', 'desc': '2x2, /2 (same)', 'out': '(24, 24, 128)'},
        {'type': 'conv', 'name': 'Conv2D', 'desc': '256 filters, 5x5, /3\n+ BatchNorm + ReLU', 'out': '(8, 8, 256)'},
        {'type': 'pool', 'name': 'MaxPool2D', 'desc': '3x3, /3 (same)', 'out': '(3, 3, 256)'},
        
        # Bloque Central
        {'type': 'conv', 'name': 'Conv2D', 'desc': '256 filters, 3x3, /1\n+ BatchNorm + ReLU', 'out': '(3, 3, 256)'},
        {'type': 'conv', 'name': 'Conv2D', 'desc': '256 filters, 1x1, /1\n+ BatchNorm + ReLU', 'out': '(3, 3, 256)'},
        {'type': 'conv', 'name': 'Conv2D', 'desc': '256 filters, 1x1, /1\n+ BatchNorm + ReLU', 'out': '(3, 3, 256)'},
        {'type': 'conv', 'name': 'Conv2D', 'desc': '256 filters, 1x1, /1\n+ BatchNorm + ReLU', 'out': '(3, 3, 256)'},
        
        {'type': 'pool', 'name': 'MaxPool2D', 'desc': '2x2, /2 (valid)', 'out': '(1, 1, 256)'},
        {'type': 'flatten', 'name': 'Flatten', 'desc': '', 'out': '(256)'},
        {'type': 'dense', 'name': 'Dense', 'desc': '512 units + ReLU', 'out': '(512)'},
        {'type': 'dropout', 'name': 'Dropout', 'desc': '0.2', 'out': '(512)'},
        {'type': 'dense', 'name': 'Dense', 'desc': '1024 units + ReLU', 'out': '(1024)'},
        {'type': 'dropout', 'name': 'Dropout', 'desc': '0.2', 'out': '(1024)'},
        
        # Capa Final
        {'type': 'dense', 'name': 'Dense (Output)', 'desc': 'Classes (Linear)', 'out': '(4)'},
    ]

    current_y = start_y

    for i, layer in enumerate(layers_data):
        # Dibujar Caja
        rect = patches.FancyBboxPatch(
            (x_center - box_width/2, current_y - box_height), 
            box_width, box_height,
            boxstyle="round,pad=0.1", 
            linewidth=1.5, 
            edgecolor='black', 
            facecolor=colors[layer['type']],
            mutation_scale=1
        )
        ax.add_patch(rect)
        
        # Texto dentro de la caja
        text_content = f"{layer['name']}"
        if layer['desc']:
            text_content += f"\n{layer['desc']}"
            
        ax.text(x_center, current_y - box_height/2, text_content, ha='center', va='center', fontsize=font_size_main, color='black')
        
        # Texto de dimensiones
        out_text = f"Salida:\n{layer['out']}"
        ax.text(x_center + box_width/2 + 0.3, current_y - box_height/2, out_text, ha='left', va='center', fontsize=font_size_dim, color='#333333')
        
        if i == len(layers_data) - 1:
             class_x = x_center
             
             # Etiqueta "Classes" debajo del rectángulo
             ax.text(class_x, current_y - box_height - 0.5, "Classes", ha='center', va='center', fontsize=14, fontweight='bold')
             
             # Lista horizontal de parámetros
             ax.text(class_x, current_y - box_height - 1.0, classes_line, ha='center', va='center', fontsize=16, color='black')

        # Flecha
        if i < len(layers_data) - 1:
            arrow_start_x = x_center
            arrow_start_y = current_y - box_height
            ax.arrow(arrow_start_x, arrow_start_y, 0, - (gap - box_height) + 0.1, head_width=0.15, head_length=0.1, fc='black', ec='black')

        current_y -= gap

    #ax.set_title("Arquitectura CNN - Lentes Gravitacionales", fontsize=18, pad=20)
    
    legend_elements = [
        patches.Patch(facecolor=colors['conv'], edgecolor='black', label='Convolución'),
        patches.Patch(facecolor=colors['pool'], edgecolor='black', label='Pooling'),
        patches.Patch(facecolor=colors['dense'], edgecolor='black', label='Densa'),
        patches.Patch(facecolor=colors['dropout'], edgecolor='black', label='Dropout')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, title="Leyenda")

    plt.tight_layout()
    plt.savefig('cnn_diagram.png', dpi=300)

draw_neural_net_diagram()
