import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def draw_node(ax, x, y, text, color='#E3F2FD', ec='#1565C0', shape='box', w=4.0, h=1.6, fontsize=10):
    # Shadow effect
    shadow_offset = 0.1
    
    # Shadow
    shadow = patches.FancyBboxPatch((x-w/2 + shadow_offset, y-h/2 - shadow_offset), w, h,
                                   boxstyle="round,pad=0.2", linewidth=0, facecolor='gray', alpha=0.3)
    ax.add_patch(shadow)
    
    # Main Box
    box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.2", 
                                 linewidth=2.5, edgecolor=ec, facecolor=color)
    ax.add_patch(box)
    
    # Text
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color='#37474F', fontfamily='sans-serif', zorder=10)

def draw_arrow(ax, x1, y1, x2, y2, text=None, rad=0.0):
    style = f"arc3,rad={rad}"
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle="-|>,head_length=0.8,head_width=0.5", 
                                    connectionstyle=style,
                                    color="#546E7A", lw=2.5, zorder=5)
    arrow.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])
    ax.add_patch(arrow)
    
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        if rad != 0:
            # Adjust label position based on curve to avoid overlap
            mid_y += rad * 1.5 
            
        t = ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=9, 
                color="#BF360C", weight='bold', backgroundcolor='white', zorder=15)
        t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

def create_overall_diagram():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Palette
    c_blue = '#E3F2FD'; ec_blue = '#1565C0'  # Data
    c_orange = '#FFF3E0'; ec_orange = '#EF6C00' # Teacher
    c_purple = '#F3E5F5'; ec_purple = '#7B1FA2' # Pseudo
    c_green = '#E8F5E9'; ec_green = '#2E7D32' # Student
    c_red = '#FFEBEE'; ec_red = '#C62828' # Inference
    
    # --- Layer 1: Data ---
    x_data = 3
    y_top = 9.5
    y_bot = 2.5
    
    draw_node(ax, x_data, y_top, "Labeled Data\n(Small)", color=c_blue, ec=ec_blue, w=3.5, h=1.4)
    draw_node(ax, x_data, y_bot, "Unlabeled Data\n(Large Test Set)", color=c_blue, ec=ec_blue, w=3.5, h=1.4)
    
    # --- Layer 2: Teacher ---
    x_teacher = 9
    y_teacher = 9.5
    
    draw_node(ax, x_teacher, y_teacher, "Teacher Model Pool\n(ConvNeXt, Swin, EVA)\n15 Models Ensemble", color=c_orange, ec=ec_orange, w=5.5, h=2.0)
    
    # --- Layer 3: Pseudo Labels ---
    x_pseudo = 9
    y_pseudo = 6.0
    
    draw_node(ax, x_pseudo, y_pseudo, "Pseudo Labeling\nConfidence > 0.99", color=c_purple, ec=ec_purple, w=4.5, h=1.5)
    
    # --- Layer 4: Student Training ---
    x_student = 16
    y_student = 6.0
    
    draw_node(ax, x_student, y_student, "Student Training\n(Noisy Student)\nData = Org + Pseudo", color=c_green, ec=ec_green, w=5.5, h=2.0)
    
    # --- Layer 5: Champion Inference ---
    x_infer = 21.5
    y_infer = 6.0
    
    draw_node(ax, x_infer, y_infer, "Champion Inference\nWeighted Voting\nThreshold Opt", color=c_red, ec=ec_red, w=4, h=1.8)
    
    # --- Arrows ---
    
    # Data -> Teacher (Straight)
    draw_arrow(ax, x_data+1.75, y_top, x_teacher-2.75, y_teacher, text="Train")
    
    # Teacher -> Pseudo (Down)
    draw_arrow(ax, x_teacher, y_teacher-1.0, x_pseudo, y_pseudo+0.75, text="Predict")
    
    # Unlabeled -> Pseudo (Curved up)
    draw_arrow(ax, x_data+1.75, y_bot, x_pseudo-2.25, y_pseudo, rad=-0.3)
    
    # Pseudo -> Student (Straight Right)
    draw_arrow(ax, x_pseudo+2.25, y_pseudo, x_student-2.75, y_pseudo, text="Expand Data")
    
    # Labeled -> Student (Original data mix in)
    # Long curved arrow from top-left to student, avoiding overlap
    # We can route it via top
    draw_arrow(ax, x_data+1.75, y_top+0.2, x_student, y_student+1.0, rad=-0.5, text="Original Data")
    
    # Student -> Inference
    draw_arrow(ax, x_student+2.75, y_student, x_infer-2.0, y_infer)

    
    plt.title("AOI Defect Classification: Teacher-Student Pipeline", fontsize=20, fontweight='bold', color='#263238', pad=20)
    plt.tight_layout()
    plt.savefig('assets/overall_architecture.png', dpi=300, bbox_inches='tight')
    print("Premium diagram generated: assets/overall_architecture.png")

if __name__ == "__main__":
    create_overall_diagram()
