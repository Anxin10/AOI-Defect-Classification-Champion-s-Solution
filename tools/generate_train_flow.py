import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def draw_box(ax, x, y, w, h, text, color='#FAFAFA', ec='#424242', fontsize=10, fontcolor='#212121', style='round,pad=0.2'):
    shadow = patches.FancyBboxPatch((x-w/2+0.08, y-h/2-0.08), w, h, boxstyle=style, linewidth=0, facecolor='black', alpha=0.15)
    ax.add_patch(shadow)
    box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle=style, linewidth=2, edgecolor=ec, facecolor=color)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=fontcolor, zorder=10)

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>,head_length=0.8,head_width=0.4', 
                                    connectionstyle="arc3,rad=0", color='#546E7A', lw=2, zorder=5)
    ax.add_patch(arrow)
    if text:
        t = ax.text((x1+x2)/2 + 0.2, (y1+y2)/2, text, fontsize=9, color='#0277BD', fontweight='bold', ha='left')
        t.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

def create_teacher_flow():
    fig, ax = plt.subplots(figsize=(10, 10)) # Reduced height
    ax.set_xlim(0, 10)
    ax.set_ylim(7, 21) # Crop bottom empty space
    ax.axis('off')
    x = 5
    
    # Title
    plt.title("Teacher Training Flow (train_teacher.py)", fontsize=18, fontweight='bold', pad=20, color='#E65100')

    # Steps
    draw_box(ax, x, 19, 6, 1.2, "Load Config & Data\n(train.csv)", color='#FFF3E0', ec='#EF6C00')
    draw_arrow(ax, x, 18.4, x, 17.6)
    
    draw_box(ax, x, 17, 6, 1.2, "5-Fold Cross Validation Split\n(StratifiedKFold)", color='#FFE0B2', ec='#FB8C00')
    draw_arrow(ax, x, 16.4, x, 15.6)
    
    # Loop
    draw_box(ax, x, 13.5, 8, 4.2, "", color='#FAFAFA', ec='#BDBDBD', style='square,pad=0.2')
    ax.text(x, 15.2, "For Each Fold (0~4)", ha='center', fontsize=12, fontweight='bold', color='#757575')
    
    draw_box(ax, x, 14.5, 6, 1.0, "Init Model (Pretrained)\nConvNeXt / Swin / EVA", color='#FFCC80', ec='#F57C00')
    draw_arrow(ax, x, 14.0, x, 13.2)
    
    draw_box(ax, x, 12.5, 6, 1.4, "Training Loop (20 Epochs)\n• AMP (FP16)\n• Grad Clipping (1.0)\n• EMA Update", color='#FFB74D', ec='#EF6C00')
    
    draw_arrow(ax, x, 11.4, x, 10.6)
    
    draw_box(ax, x, 10, 6, 1.2, "Save Best Model\n(foldX_best.pth)", color='#FF9800', ec='#E65100')
    draw_arrow(ax, x, 9.4, x, 8.6)
    
    draw_box(ax, x, 8, 6, 1.2, "Save Last Checkpoint\n(Ensure Auto-Resume)", color='#F57C00', ec='#E65100', fontcolor='white')

    plt.tight_layout()
    plt.savefig('assets/teacher_flow.png', dpi=300, bbox_inches='tight')
    print("Generated: assets/teacher_flow.png")

def create_student_flow():
    fig, ax = plt.subplots(figsize=(10, 10)) # Reduced height
    ax.set_xlim(0, 10)
    ax.set_ylim(7, 21) # Crop bottom empty space
    ax.axis('off')
    x = 5
    
    # Title
    plt.title("Student Training Flow (train_student.py)", fontsize=18, fontweight='bold', pad=20, color='#2E7D32')

    # Steps
    draw_box(ax, x, 19, 6, 1.2, "Load Original + Pseudo Data", color='#E8F5E9', ec='#2E7D32')
    draw_arrow(ax, x, 18.4, x, 17.6)
    
    draw_box(ax, x, 17, 7, 1.2, "Dataset concatenation\n(Train size: 2528 -> 12000+)", color='#C8E6C9', ec='#43A047')
    draw_arrow(ax, x, 16.4, x, 15.6)
    
    # Loop
    draw_box(ax, x, 13.5, 8, 4.2, "", color='#FAFAFA', ec='#BDBDBD', style='square,pad=0.2')
    ax.text(x, 15.2, "For Each Fold (0~4)", ha='center', fontsize=12, fontweight='bold', color='#757575')
    
    draw_box(ax, x, 14.5, 6, 1.0, "Init Noisy Student Model\n(Stronger Augmentations)", color='#A5D6A7', ec='#2E7D32')
    draw_arrow(ax, x, 14.0, x, 13.2)
    
    draw_box(ax, x, 12.5, 6, 1.4, "Student Training Loop\n• Learn from Noisy Data\n• Larger Batch Size\n• Harder Regularization", color='#81C784', ec='#2E7D32')
    
    draw_arrow(ax, x, 11.4, x, 10.6)
    
    draw_box(ax, x, 10, 6, 1.2, "Save Best Student\n(student_foldX_best.pth)", color='#66BB6A', ec='#1B5E20')
    draw_arrow(ax, x, 9.4, x, 8.6)
    
    draw_box(ax, x, 8, 6, 1.2, "Save Last Checkpoint", color='#4CAF50', ec='#1B5E20', fontcolor='white')

    plt.tight_layout()
    plt.savefig('assets/student_flow.png', dpi=300, bbox_inches='tight')
    print("Generated: assets/student_flow.png")

if __name__ == "__main__":
    create_teacher_flow()
    create_student_flow()
