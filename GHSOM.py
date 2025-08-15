import os
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from CORE_CLASS.GHSOM_Train import GHSOMTrain
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import numpy as np
import networkx as nx
import pandas as pd



class GHSOM:
    def __init__(self, tau_1=0.1, tau_2=0.3, max_depth=3, initial_map_size=(2, 2), tracking=2):
        """
        Initialize the Growing Hierarchical Self-Organizing Map (GHSOM).

        Parameters:
        ----------
        tau_1 : float, default=0.1
            Threshold for vertical growth (controls hierarchical depth)
        tau_2 : float, default=0.2
            Threshold for horizontal growth (controls map size at each level)
        max_depth : int, default=3
            Maximum depth of the hierarchy
        initial_map_size : tuple, default=(2, 2)
            Initial size of the root map
        tracking : int, default=0
            Level of training progress tracking (0=none, 1=minimal, 2=detailed)
        """
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.max_depth = max_depth
        self.initial_map_size = initial_map_size
        self.tracking = tracking
        # only draw viz on the very first train() call
        self._did_visualize = False

    def train(self, data, labels=None):
        """
        Train the GHSOM with the given data.

        Parameters:
        ----------
        data : numpy.ndarray
            Training data
        labels : list, optional
            Labels for the training data

        Returns:
        -------
        ghsom_structure : dict
            Trained GHSOM structure
        """
        trainer = GHSOMTrain(
            data=data,
            labels=labels,
            tracking=self.tracking
        )

        trainer.depth = self.tau_1
        trainer.breadth = self.tau_2
        trainer.max_depth = self.max_depth

        ghsom_structure = trainer.train_ghsom(
            data=data,
            labels=labels,
            initial_map_size=self.initial_map_size,
            tau_1=self.tau_1,
            tau_2=self.tau_2,
            max_depth=self.max_depth
        )

        # Root BMUs
        root = ghsom_structure['root_map']
        bmus_root, _ = trainer._calculate_bmus_and_qerr(root['codebook'], data)
        root['bmus'] = bmus_root
        root['labels'] = list(labels)

        # Child BMUs
        for unit_idx, child in enumerate(root.get('children', [])):
            if child is None:
                continue
            mask = (bmus_root == unit_idx)
            sub_data = data[mask]
            sub_labels = np.array(labels)[mask].tolist()
            bmus_sub, _ = trainer._calculate_bmus_and_qerr(child['codebook'], sub_data)
            child['bmus'] = bmus_sub
            child['labels'] = sub_labels

        # --- Only visualize once ---
        if not self._did_visualize:
            os.makedirs("Results", exist_ok=True)
            self.visualize_ghsom(root)
            self.ghsom_grid(root)
            self.visualize_component_planes(root, data)
            self.ghsom_tree(root)
            self.ghsom_tree_3d(root)
            self._did_visualize = True

        return ghsom_structure

    def visualize_ghsom(self, ghsom, level=0, parent_label=None):
        """
        Visualize the GHSOM structure, showing the hierarchy of maps.

        Parameters:
        ----------
        ghsom : dict
            The trained GHSOM structure
        level : int, default=0
            Current hierarchy level for recursive visualization
        parent_label : str, optional
            Label of the parent unit
        """
        if level == 0:
            # Create a new figure for the root level
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.title('GHSOM Structure')
        else:
            # Get current figure for child levels
            fig = plt.gcf()
            ax = plt.gca()

        # Get map dimensions
        msize = ghsom['topol']['msize']

        # Calculate the position of this map based on level and parent
        if level == 0:
            # Root map is centered
            base_x = 0
            base_y = 0
            width = 10
            height = 10
        else:
            # Position child maps relative to parent
            # These calculations depend on the parent_label format
            parts = parent_label.split('_')
            parent_x = float(parts[-2])
            parent_y = float(parts[-1])

            # Calculate size based on level (smaller as we go deeper)
            scale_factor = 0.7 ** level
            width = 10 * scale_factor
            height = 10 * scale_factor

            # Position relative to parent
            base_x = parent_x
            base_y = parent_y

        # Draw the map grid
        cell_width = width / msize[1]
        cell_height = height / msize[0]

        # Draw each unit in the map
        for i in range(msize[0]):
            for j in range(msize[1]):
                unit_idx = i * msize[1] + j

                # Calculate unit position
                x = base_x + j * cell_width
                y = base_y + i * cell_height

                # Draw unit rectangle
                rect = Rectangle((x, y), cell_width, cell_height,
                                 fill=False, edgecolor='black', linewidth=1)
                ax.add_patch(rect)

                # Label for this unit
                if level == 0:
                    unit_label = f"{unit_idx}"
                else:
                    unit_label = f"{parent_label}_{unit_idx}"

                # Add unit label
                ax.text(x + cell_width / 2, y + cell_height / 2, unit_label,
                        ha='center', va='center', fontsize=8)

                # Check if this unit has a child map
                if 'children' in ghsom and unit_idx < len(ghsom['children']) and ghsom['children'][
                    unit_idx] is not None:
                    # Add indicator that this unit has a child
                    ax.text(x + cell_width / 2, y + cell_height * 0.8, '↓',
                            ha='center', va='center', fontsize=10, color='red')

                    # Recursively visualize the child map
                    child_label = f"{unit_label}_{x + cell_width / 2}_{y + cell_height * 1.5}"
                    self.visualize_ghsom(ghsom['children'][unit_idx], level + 1, child_label)

        # Set axis properties for the root level
        if level == 0:
            ax.set_xlim(-5, 15)
            ax.set_ylim(-5, 15)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save to Results folder
            save_path = os.path.join("Results", "Visualize_ghsom.png")
            fig.savefig(save_path)

            return fig

    def ghsom_grid(self, ghsom, max_cols=None, save_path="Results/ghsom_grid.pdf"):
        """
        Draw each SOM map in its own small subplot, arranged by hierarchy level.
        One row per level, one column per map in that level.
        """
        # --- 1) collect all maps by depth ---
        maps_by_level = defaultdict(list)

        def collect(node, path, level):
            label = ".".join(path)
            maps_by_level[level].append((node['topol']['msize'], label))
            for idx, child in enumerate(node.get('children', [])):
                if child is not None:
                    collect(child, path + [str(idx)], level + 1)

        collect(ghsom, ['root'], 0)
        n_levels = max(maps_by_level) + 1

        # auto-compute columns or cap to max_cols if given
        cols = max(len(maps_by_level[lvl]) for lvl in range(n_levels))
        if max_cols is not None:
            cols = min(cols, max_cols)

        # --- 2) build figure/grid ---
        fig, axes = plt.subplots(n_levels, cols,
                                 figsize=(cols * 2, n_levels * 2),
                                 squeeze=False)
        for lvl in range(n_levels):
            maps = maps_by_level[lvl]
            for col in range(cols):
                ax = axes[lvl][col]
                if col < len(maps):
                    msize, label = maps[col]
                    nrows, ncols = msize
                    cw, ch = 1.0 / ncols, 1.0 / nrows

                    # draw each cell
                    for i in range(nrows):
                        for j in range(ncols):
                            rect = Rectangle((j * cw, 1 - (i + 1) * ch),
                                             cw, ch,
                                             fill=False, edgecolor='black', lw=0.5)
                            ax.add_patch(rect)
                            ax.text(j * cw + cw / 2,
                                    1 - (i + 0.5) * ch,
                                    str(i * ncols + j),
                                    ha='center', va='center',
                                    fontsize=6)

                    ax.set_title(label, fontsize=8)
                    ax.set_xticks([]);
                    ax.set_yticks([])
                    ax.set_xlim(0, 1);
                    ax.set_ylim(0, 1)
                else:
                    ax.axis('off')

            # label the row once on the left
            axes[lvl][0].set_ylabel(f"Level {lvl}", rotation=0, labelpad=30, va='center')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        return fig

    def ghsom_tree(self, ghsom_root, vert_gap=0.25):
        """
        Draws the GH-SOM hierarchy as a directed tree of circles labeled by map size.
        Dynamically sizes the figure to keep the bottom row readable.
        Saves to Results/ghsom.png.
        """
        # ensure Results dir exists
        os.makedirs("Results", exist_ok=True)

        # 1) Traverse and build graph, recording depths
        G = nx.DiGraph()
        depths = {}

        def traverse(node, name="root", depth=0):
            depths[name] = depth
            nrows, ncols = node['topol']['msize']
            label = f"{name}\n[{nrows}×{ncols}]"
            G.add_node(name, label=label)
            for idx, child in enumerate(node.get('children', [])):
                if child is None:
                    continue
                child_name = f"{name}.{idx}"
                G.add_edge(name, child_name)
                traverse(child, child_name, depth + 1)

        traverse(ghsom_root)
        max_depth = max(depths.values())
        bottom_count = sum(1 for d in depths.values() if d == max_depth)

        # 2) Dynamic figure sizing
        fig_width = max(12, bottom_count * 0.5)  # 0.5″ per bottom-level node, min 12″
        fig_height = (max_depth + 2) * 1.2  # 1.2″ per level

        # 3) Compute hierarchy positions
        def hierarchy_pos(G, root="root", width=1., vert_loc=1., xcenter=0.5):
            def _hierarchy_pos(G, r, left, right, vert_loc, xcenter, pos):
                pos[r] = (xcenter, vert_loc)
                children = list(G.successors(r))
                if children:
                    dx = (right - left) / len(children)
                    nextx = left + dx / 2
                    for c in children:
                        pos = _hierarchy_pos(
                            G, c,
                            nextx - dx / 2, nextx + dx / 2,
                            vert_loc - vert_gap, nextx, pos
                        )
                        nextx += dx
                return pos

            return _hierarchy_pos(G, root, 0, width, 1.0, 0.5, {})

        pos = hierarchy_pos(G)

        # 4) Draw
        fig = plt.figure(figsize=(fig_width, fig_height))
        nx.draw(
            G, pos,
            labels=nx.get_node_attributes(G, 'label'),
            with_labels=True,
            node_size=500,
            node_color='white',
            edgecolors='black',
            arrows=True,
            font_size=6
        )
        plt.axis('off')
        plt.tight_layout()

        # 5) Save to Results folder
        save_path = os.path.join("Results", "ghsom_tree.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved GH-SOM tree to {save_path}")


    def calculate_quantization_error(self, ghsom, data, return_by_level=False):
        """
        Calculate the quantization error of the GHSOM for the given data.

        Parameters:
        ----------
        ghsom : dict
            The trained GHSOM structure
        data : numpy.ndarray
            Data to calculate quantization error for
        return_by_level : bool, default=False
            Whether to return errors by hierarchy level

        Returns:
        -------
        qerror : float or dict
            Mean quantization error, or dict of errors by level if return_by_level=True
        """
        # Create a trainer for utility functions
        trainer = GHSOMTrain(data=data, tracking=0)

        # Use recursive function to calculate errors through the hierarchy
        all_errors, data_counts = self._calculate_qerror_recursive(ghsom, data, trainer, level=0)

        # Calculate weighted average across all levels
        total_error = 0
        total_count = 0
        for level, level_errors in all_errors.items():
            level_count = data_counts[level]
            if level_count > 0:
                total_error += sum(level_errors)
                total_count += level_count

        if return_by_level:
            # Convert lists of errors to mean errors by level
            mean_errors_by_level = {}
            for level, level_errors in all_errors.items():
                if data_counts[level] > 0:
                    mean_errors_by_level[level] = sum(level_errors) / data_counts[level]
                else:
                    mean_errors_by_level[level] = 0
            return mean_errors_by_level
        else:
            # Return the overall mean quantization error
            return total_error / total_count if total_count > 0 else 0

    def _calculate_qerror_recursive(self, ghsom, data, trainer, level=0):
        """
        Recursively calculate quantization errors through all hierarchy levels.

        Parameters:
        ----------
        ghsom : dict
            The trained GHSOM structure
        data : numpy.ndarray
            Data to calculate quantization error for
        trainer : GHSOMTrain
            Trainer instance for utility functions
        level : int
            Current hierarchy level

        Returns:
        -------
        all_errors : dict
            Dictionary of errors by level
        data_counts : dict
            Dictionary of data counts by level
        """
        # Initialize error and count dictionaries
        all_errors = {level: []}
        data_counts = {level: 0}

        # Calculate BMUs and quantization errors for this map
        bmus, qerrors = trainer._calculate_bmus_and_qerr(ghsom['codebook'], data)
        all_errors[level].extend(qerrors)
        data_counts[level] = len(data)

        # Check if this map has children
        if 'children' in ghsom and any(child is not None for child in ghsom['children']):
            # For each child map, find the data that maps to its parent unit
            for unit_idx, child in enumerate(ghsom['children']):
                if child is not None:
                    # Find data mapped to this unit
                    unit_data_indices = np.where(bmus == unit_idx)[0]
                    if len(unit_data_indices) > 0:
                        unit_data = data[unit_data_indices]

                        # Recursively calculate errors for the child map
                        child_errors, child_counts = self._calculate_qerror_recursive(
                            child, unit_data, trainer, level + 1)

                        # Merge results
                        for child_level, errors in child_errors.items():
                            if child_level not in all_errors:
                                all_errors[child_level] = []
                            all_errors[child_level].extend(errors)

                            if child_level not in data_counts:
                                data_counts[child_level] = 0
                            data_counts[child_level] += child_counts[child_level]

        return all_errors, data_counts

    def visualize_component_planes(self, ghsom, data, feature_names=None, level=0, parent_label=None, figsize=(15, 10)):
        """
        Visualize component planes of a GHSOM map showing how each feature is distributed.

        Parameters:
        ----------
        ghsom : dict
            The trained GHSOM structure
        data : numpy.ndarray
            Data used for training the GHSOM
        feature_names : list, optional
            Names of features to use as titles
        level : int, default=0
            Current hierarchy level (used for recursive visualization)
        parent_label : str, optional
            Label of parent unit (used for recursive visualization)
        figsize : tuple, default=(15, 10)
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(data.shape[1])]

        if level == 0:
            # Create a new figure for root level component planes
            fig = plt.figure(figsize=figsize)
            n_features = data.shape[1]

            # Calculate grid dimensions for subplots
            grid_size = int(np.ceil(np.sqrt(n_features)))

            # Get map dimensions and codebook
            msize = ghsom['topol']['msize']
            codebook = ghsom['codebook']

            # Create a subplot for each feature
            for i in range(n_features):
                ax = plt.subplot(grid_size, grid_size, i + 1)

                # Extract feature values from codebook
                feature_values = codebook[:, i].reshape(msize)

                # Create a heatmap for this feature
                norm = Normalize(vmin=np.min(feature_values), vmax=np.max(feature_values))
                cmap = cm.viridis

                # Plot the heatmap
                im = ax.imshow(feature_values, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')

                # Add a colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Set title and turn off axis ticks
                ax.set_title(feature_names[i], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

                # Add unit indices as text
                for y in range(msize[0]):
                    for x in range(msize[1]):
                        unit_idx = y * msize[1] + x
                        ax.text(x, y, str(unit_idx), ha='center', va='center',
                                color='white' if feature_values[y, x] > np.mean(feature_values) else 'black',
                                fontsize=8)

                # Check if this unit has a child map
                if 'children' in ghsom:
                    for unit_idx in range(len(ghsom['children'])):
                        if ghsom['children'][unit_idx] is not None:
                            # Calculate coordinates
                            y, x = divmod(unit_idx, msize[1])
                            # Add a marker to indicate there's a submap
                            ax.plot(x, y, 'rs', markersize=5, alpha=0.7)

            plt.tight_layout()

            # You can also recursively visualize child maps' component planes
            # if desired by adding code similar to the visualize_ghsom method

            # Save to Results folder
            save_path = os.path.join("Results", "Visualize_component_planes.png")
            fig.savefig(save_path)

            return fig

    def draw_layered_map(self, ax, msize, color_grid, origin,
                         scale=0.4, skew=0.3, cmap=None, empty_alpha=0.2):
        """
        Draw one SOM map as a skewed plate of colored circles, with a tilted border.
        Returns: xmin, xmax, ymin, ymax, plate_corners.
        """
        if cmap is None:
            cmap = cm.get_cmap('tab20')

        nrows, ncols = msize
        xs, ys = [], []
        # corners of the tilted plate
        bl = (origin[0], origin[1] + 0 * skew)
        br = (origin[0] + ncols * scale, origin[1] + 0 * skew)
        tr = (origin[0] + ncols * scale, origin[1] + nrows * scale + nrows * skew)
        tl = (origin[0], origin[1] + nrows * scale + nrows * skew)
        plate_corners = [bl, br, tr, tl]

        # draw circles
        for i in range(nrows):
            for j in range(ncols):
                cx = origin[0] + j * scale + scale / 2
                cy = origin[1] + i * scale + i * skew + scale / 2
                idx = color_grid[i, j]
                if idx >= 0:
                    face = cmap(int(idx) % cmap.N)
                    edge = 'k';
                    alpha = 0.9
                else:
                    face = 'white'
                    edge = 'lightgray';
                    alpha = empty_alpha

                circ = Circle((cx, cy),
                              radius=scale * 0.45,
                              facecolor=face,
                              edgecolor=edge,
                              alpha=alpha,
                              linewidth=1)
                ax.add_patch(circ)
                xs += [cx - scale / 2, cx + scale / 2]
                ys += [cy - scale / 2, cy + scale / 2]

        # draw plate border
        border = Polygon(plate_corners, facecolor='none', edgecolor='gray', linewidth=1)
        ax.add_patch(border)

        return min(xs), max(xs), min(ys), max(ys), plate_corners

    def ghsom_tree_3d(self, root,
                      child_spacing=3.0,
                      v_gap=1.0,
                      scale=0.4,
                      skew=0.3):
        """
        GH-SOM tree with two separation lines:
         - “Layer 1” just below the root
         - “Layer 2” just below the children
        """


        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = cm.get_cmap('tab20')

        # ─── DRAW ROOT PLATE ────────────────────────────────────────
        nrows, ncols = root['topol']['msize']
        grid_root = np.arange(nrows * ncols).reshape(nrows, ncols)
        xr = - (ncols * scale) / 2
        yr = 0
        xmin_r, xmax_r, ymin_r, ymax_r, _ = self.draw_layered_map(
            ax, (nrows, ncols), grid_root,
            origin=(xr, yr),
            scale=scale, skew=skew, cmap=cmap
        )

        # ─── DRAW CHILD PLATES ───────────────────────────────────────
        children = [c for c in root.get('children', []) if c]
        m = len(children)
        span = (m - 1) * child_spacing
        x_centers = [-span / 2 + i * child_spacing for i in range(m)]
        y_child = ymin_r - v_gap - (nrows * scale + nrows * skew)

        child_extents = []
        for idx, child in enumerate(children):
            nr, nc = child['topol']['msize']
            grid_c = np.full((nr, nc), idx, dtype=int)
            y_child = ymin_r - 2 * v_gap - (nr * scale + nr * skew)
            x0 = x_centers[idx] - (nc * scale) / 2

            xmin_c, xmax_c, ymin_c, ymax_c, _ = self.draw_layered_map(
                ax, (nr, nc), grid_c,
                origin=(x0, y_child),
                scale=scale, skew=skew, cmap=cmap
            )
            child_extents.append((xmin_c, xmax_c, ymin_c, ymax_c))

            # arrow from root cell to this child
            pr, pc = divmod(idx, ncols)
            sx = xr + pc * scale + scale / 2
            sy = yr + pr * scale + pr * skew + scale / 2
            cx = x0 + (nc * scale) / 2
            cy = y_child + (nr * scale + nr * skew) / 2
            ax.add_patch(FancyArrowPatch((sx, sy), (cx, cy),
                                         connectionstyle="arc3,rad=0.4",
                                         arrowstyle='->',
                                         lw=1,
                                         color='gray'))

            # 3) GLOBAL X BOUNDS
        all_x = [xmin_r, xmax_r] + [v for e in child_extents for v in (e[0], e[1])]
        global_xmin, global_xmax = min(all_x) - child_spacing, max(all_x) + child_spacing

        # 4) LAYER SEPARATORS & LABELS (draw last, on top)
        # how much to nudge down (10% of your v_gap)
        delta = 0.1 * v_gap

        # Layer 1: just below root
        y1 = ymin_r - delta
        ax.hlines(y1, global_xmin, global_xmax, color='gray', lw=1, zorder=10)
        ax.text(global_xmax, y1, "Layer 1", ha='left', va='center', fontweight='bold', zorder=11)

        # Layer 2: just below children
        ymin_ch = min(e[2] for e in child_extents)
        y2 = ymin_ch - v_gap / 2
        ax.hlines(y2, global_xmin, global_xmax,
                  color='gray', lw=1, zorder=10)
        ax.text(global_xmax, y2,
                "Layer 2", ha='left', va='center',
                fontweight='bold', zorder=11)

        # 5) FINALIZE & SAVE
        # Collect all y’s for framing
        ys = [ymax_r, ymin_r, y1, y2] + [val for e in child_extents for val in (e[2], e[3])]
        y_min, y_max = min(ys), max(ys)
        pad = v_gap * 1.5

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        os.makedirs("Results", exist_ok=True)
        fig.savefig("Results/ghsom_tree_3d_final_neat.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

