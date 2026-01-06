import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def draw_box(ax, x, y, w, h, text, color='#FAFAFA', ec='#424242', fontsize=10, fontcolor='#212121', style='round,pad=0.2'):
    # Shadow
    shadow = patches.FancyBboxPatch((x-w/2+0.08, y-h/2-0.08), w, h, boxstyle=style, linewidth=0, facecolor='black', alpha=0.15)
    ax.add_patch(shadow)
    # Box
    box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=style, linewidth=2, edgecolor=ec, facecolor=color)
    ax.add_patch(box)
    # Text
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=fontcolor, zorder=10)

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>,head_length=0.8,head_width=0.4', 
                                    connectionstyle="arc3,rad=0", color='#546E7A', lw=2, zorder=5)
    ax.add_patch(arrow)
    if text:
        t = ax.text((x1+x2)/2 + 0.2, (y1+y2)/2, text, fontsize=9, color='#0277BD', fontweight='bold', ha='left')
        t.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

def create_code_flow():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    x_center = 5
    
    # 1. Multi-Models (Top Layer)
    y_models = 18
    # Draw separate boxes for models to show "Multi-Model" concept
    draw_box(ax, 2.5, y_models, 2.5, 1.2, "ConvNeXt V2\n(Large)", color='#E3F2FD', ec='#1565C0')
    draw_box(ax, 5.0, y_models, 2.0, 1.2, "Swin V2\n(Large)", color='#E3F2FD', ec='#1565C0')
    draw_box(ax, 7.5, y_models, 2.0, 1.2, "EVA-02\n(Large)", color='#E3F2FD', ec='#1565C0')
    
    # 2. Intra-Model Processing (The Core Engine)
    y_engine = 12.5
    # Big container box
    draw_box(ax, x_center, y_engine, 8.5, 7.0, "", color='#FAFAFA', ec='#9E9E9E', style='square,pad=0.2')
    
    # Title for container
    ax.text(x_center, y_engine+2.8, "Intra-Model Optimization (Inside Each Model)", 
            ha='center', fontsize=12, fontweight='bold', color='#616161')
            
    # Sub-steps inside
    draw_box(ax, x_center, y_engine+1.5, 6, 1.2, "5-Fold Cross Validation\n(5 Models per Architecture)", color='#FFF3E0', ec='#EF6C00')
    draw_box(ax, x_center, y_engine-0.5, 6, 1.2, "5-View Parallel TTA\n(Original + Flips + Rot90)", color='#FFF3E0', ec='#EF6C00')
    draw_box(ax, x_center, y_engine-2.5, 6, 1.2, "Power Mean (p=1.5)\n(Combine 5 Folds & 5 Views)", color='#FFF3E0', ec='#EF6C00')
    
    # Arrows inside
    draw_arrow(ax, x_center, y_engine+0.9, x_center, y_engine+0.1)
    draw_arrow(ax, x_center, y_engine-1.1, x_center, y_engine-1.9)
    
    # Arrows from Top Models to Container
    draw_arrow(ax, 2.5, y_models-0.6, 2.5, y_engine+3.5)
    draw_arrow(ax, 5.0, y_models-0.6, 5.0, y_engine+3.5)
    draw_arrow(ax, 7.5, y_models-0.6, 7.5, y_engine+3.5)
    
    # 3. Weighted Ensemble
    y_ensemble = 7
    draw_box(ax, x_center, y_ensemble, 7, 1.5, "Weighted Ensemble Strategy\nConvNeXt(0.5) + Swin(0.3) + EVA(0.2)", color='#F3E5F5', ec='#7B1FA2')
    draw_arrow(ax, x_center, y_engine-3.5, x_center, y_ensemble+0.75)
    
    # 4. Post-Processing
    y_post = 4
    draw_box(ax, x_center, y_post, 6, 1.5, "Artificial Post-Processing\n(Rare Class Rescue)\nIf Label 2 Conf > 0.4 → Force Label 2", color='#E8F5E9', ec='#2E7D32')
    draw_arrow(ax, x_center, y_ensemble-0.75, x_center, y_post+0.75)
    
    # 5. Submission
    y_end = 1.5
    draw_box(ax, x_center, y_end, 4, 1.0, "Submission.csv", color='#212121', fontcolor='white')
    draw_arrow(ax, x_center, y_post-0.75, x_center, y_end+0.5)
    
    # Title
    plt.title("Winning Inference Architecture Flow", fontsize=20, fontweight='bold', pad=20, color='#37474F')
    
    plt.tight_layout()
    plt.savefig('assets/code_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("Generated: assets/code_flow_diagram.png")

if __name__ == "__main__":
    create_code_flow()
