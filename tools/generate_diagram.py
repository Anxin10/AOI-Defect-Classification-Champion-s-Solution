import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def draw_node(ax, x, y, text, color='#E3F2FD', ec='#1565C0', shape='box', w=3.5, h=1.2):
    # Shadow effect
    shadow_offset = 0.1
    if shape == 'box':
        # Shadow
        shadow = patches.FancyBboxPatch((x-w/2 + shadow_offset, y-h/2 - shadow_offset), w, h,
                                       boxstyle="round,pad=0.2", linewidth=0, facecolor='gray', alpha=0.3)
        ax.add_patch(shadow)
        
        # Main Box
        box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.2", 
                                     linewidth=2.5, edgecolor=ec, facecolor=color)
        ax.add_patch(box)
    
    # Text with slight stroke for readability if needed, but clean is better
    ax.text(x, y, text, ha='center', va='center', fontsize=11, 
            fontweight='bold', color='#37474F', fontfamily='sans-serif', zorder=10)

def draw_arrow(ax, x1, y1, x2, y2, text=None, rad=0.0):
    # Connect slightly off-center to avoid overlapping box edges if they were close, 
    # but here coordinates are spaced out.
    # Adjust start/end to be at box edges roughly
    
    style = f"arc3,rad={rad}"
    
    # Arrow with border
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle="-|>,head_length=0.8,head_width=0.5", 
                                    connectionstyle=style,
                                    color="#546E7A", lw=2.5, zorder=5)
    
    # Add white outline to arrow to separate from background/noise
    arrow.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])
    ax.add_patch(arrow)
    
    if text:
        # Calculate mid point for curve
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        if rad != 0:
            mid_y += rad * 2 # Approximate offset for curve label
            
        t = ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=10, 
                color="#BF360C", weight='bold', backgroundcolor='white', zorder=15)
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

def create_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define cleaner layout coordinates
    x_input = 2.5
    x_switch = 7.5
    x_aug = 12.5
    x_model = 17.5
    x_out = 21.0 # This might be too far, let's adjust
    
    # Re-centering
    x_input = 3
    x_switch = 8
    x_aug_blur = 13
    x_model = 18
    # Actually let's make it linear flow
    
    # Redefine for better flow
    x1, x2, x3, x4 = 3, 9, 15, 20
    y_mid = 5
    y_high = 8
    y_low = 2
    
    # Palette
    c_blue = '#E1F5FE'; ec_blue = '#0288D1'
    c_orange = '#FFF3E0'; ec_orange = '#F57C00'
    c_green = '#E8F5E9'; ec_green = '#388E3C'
    c_purple = '#F3E5F5'; ec_purple = '#7B1FA2'
    
    # Nodes
    draw_node(ax, x1, y_mid, "Input Image", color=c_blue, ec=ec_blue)
    
    draw_node(ax, x2, y_mid, "Augmentation\nSwitch\n(Random)", color=c_orange, ec=ec_orange, shape='box')
    
    draw_node(ax, x3, y_high, "Blur\n(Shape Focus)", color=c_purple, ec=ec_purple)
    draw_node(ax, x3, y_low, "Sharpen\n(Texture Focus)", color=c_purple, ec=ec_purple)
    
    draw_node(ax, x1 + 16, y_mid, "Single Model\n(One-Stream)", color=c_blue, ec=ec_blue)
    
    # Let's add a final node? The user graph had Robust Features
    # Adjust layout to fit 5 steps horizontally
    # Let's shift everything left slightly
    
    x1, x2, x3, x4, x5 = 2.5, 7, 11.5, 16, 20.5
    
    ax.clear()
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    draw_node(ax, x1, y_mid, "Input Image", color=c_blue, ec=ec_blue)
    draw_node(ax, x2, y_mid, "Augmentation\nSwitch", color=c_orange, ec=ec_orange)
    
    draw_node(ax, x3, y_high, "Blur\n(Shape Focus)", color=c_purple, ec=ec_purple)
    draw_node(ax, x3, y_low, "Sharpen\n(Texture Focus)", color=c_purple, ec=ec_purple)
    
    draw_node(ax, x4, y_mid, "Single Model\n(One-Stream)", color=c_blue, ec=ec_blue)
    draw_node(ax, x5, y_mid, "Robust\nFeatures", color=c_green, ec=ec_green)
    
    # Arrows
    # 1 -> 2
    draw_arrow(ax, x1+1.8, y_mid, x2-1.8, y_mid)
    
    # 2 -> 3 (Switch to Top)
    # Using curve
    draw_arrow(ax, x2+1.0, y_mid+0.6, x3-1.8, y_high, rad=-0.3, text="Epoch N")
    
    # 2 -> 3 (Switch to Bot)
    draw_arrow(ax, x2+1.0, y_mid-0.6, x3-1.8, y_low, rad=0.3, text="Epoch N+1")
    
    # 3 (Top) -> 4
    draw_arrow(ax, x3+1.8, y_high, x4-1.0, y_mid+0.6, rad=-0.3)
    
    # 3 (Bot) -> 4
    draw_arrow(ax, x3+1.8, y_low, x4-1.0, y_mid-0.6, rad=0.3)
    
    # 4 -> 5
    draw_arrow(ax, x4+1.8, y_mid, x5-1.8, y_mid)
    
    # Title
    plt.title("Dual Stream Simulation Strategy", fontsize=18, fontweight='bold', color='#263238', pad=10)
    
    plt.tight_layout()
    plt.savefig('dual_stream_sim.png', dpi=300, bbox_inches='tight')
    print("Premium diagram generated: dual_stream_sim.png")

if __name__ == "__main__":
    create_diagram()
