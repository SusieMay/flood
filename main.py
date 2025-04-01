import numpy as np
from PIL import Image
import random
import pygame
import sys

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
CELL_SIZE = 1
FPS = 60
SLIDER_HEIGHT = 40
SLIDER_MARGIN = 20
SLIDER_COLOR = (100, 100, 100, 180)
SLIDER_HANDLE_COLOR = (200, 200, 200)
SLIDER_HANDLE_WIDTH = 10
SLIDER_HANDLE_HEIGHT = 30
MAX_STEPS = 1

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (185, 122, 87)
CHART_BG = (30, 30, 30)
GRID_COLOR = (50, 50, 50)

def load_map(image_path, satellite_path=None):
    # Wczytaj podstawową mapę
    main_image = Image.open(image_path).convert('RGB')
    satellite_array = None

    # Wczytaj zdjęcie satelitarne jeśli podano
    if satellite_path:
        satellite_image = Image.open(satellite_path).convert('RGB')
        satellite_array = np.array(satellite_image)
        if satellite_array.shape[2] == 3:
            satellite_array = np.transpose(satellite_array, (1, 0, 2))

    height, width = main_image.size[1], main_image.size[0]
    data = np.array(main_image)

    terrain_map = np.zeros((height, width))
    water_depth = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            r, g, b = data[y, x]
            if (r, g, b) == (0, 0, 0):
                terrain_map[y, x] = 5
            elif (r, g, b) == (185, 122, 87):
                terrain_map[y, x] = 4
            elif (r, g, b) == (28, 32, 88):
                terrain_map[y, x] = 3
                water_depth[y, x] = 1.0
            else:
                terrain_map[y, x] = 0

    return terrain_map, water_depth, satellite_array

def get_water_color(depth):
    ratio = min(max(depth, 0), 1.0)
    r = int(255 * (1 - ratio))
    g = int(255 * (1 - ratio))
    b = int(255 - (100 * (1 - ratio)))
    return (r, g, b)

def get_flooded_terrain_color():
    return (100, 80, 90)

def has_empty_or_flooded_neighbors(x, y, terrain):
    neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
    for ny, nx in neighbors:
        if 0 <= ny < terrain.shape[0] and 0 <= nx < terrain.shape[1]:
            if terrain[ny, nx] == 0 or terrain[ny, nx] == 6:
                return True
    return False

def get_adjacent_water_cells(x, y, terrain):
    neighbors = []
    for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
        if 0 <= ny < terrain.shape[0] and 0 <= nx < terrain.shape[1]:
            if terrain[ny, nx] in [3, 6]:
                neighbors.append((ny, nx))
    return neighbors

def check_flood_terrain(x, y, terrain, water_depth):
    water_neighbors = []
    neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]

    for ny, nx in neighbors:
        if 0 <= ny < terrain.shape[0] and 0 <= nx < terrain.shape[1]:
            if terrain[ny, nx] in [3, 6]:
                water_neighbors.append(water_depth[ny, nx])

    if water_neighbors and min(water_neighbors) >= 0.98:
        return True
    return False

def update_cell(x, y, terrain, water_depth, new_terrain, new_water_depth, flow_speed):
    if terrain[y, x] in [3, 6]:
        current_depth = water_depth[y, x]

        if has_empty_or_flooded_neighbors(x, y, terrain):
            neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            random.shuffle(neighbors)

            depth_decrease = 0.05 * flow_speed

            for ny, nx in neighbors:
                if 0 <= ny < terrain.shape[0] and 0 <= nx < terrain.shape[1]:
                    if terrain[ny, nx] == 0:
                        new_terrain[ny, nx] = 3
                        new_water_depth[ny, nx] = max(0.2, current_depth - depth_decrease)
                        break
                    elif terrain[ny, nx] == 6:
                        new_water_depth[ny, nx] = max(new_water_depth[ny, nx],min(1.0, current_depth - depth_decrease))
                        break

        water_neighbors = get_adjacent_water_cells(x, y, terrain)
        if water_neighbors:
            avg_neighbor_depth = sum(water_depth[ny, nx] for ny, nx in water_neighbors) / len(water_neighbors)
            new_water_depth[y, x] = min(1.0, max(current_depth, avg_neighbor_depth + (0.01 * flow_speed)))

    elif terrain[y, x] == 4 and check_flood_terrain(x, y, terrain, water_depth):
        new_terrain[y, x] = 6
        new_water_depth[y, x] = 0.2

        neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        for ny, nx in neighbors:
            if 0 <= ny < terrain.shape[0] and 0 <= nx < terrain.shape[1]:
                if terrain[ny, nx] == 4:
                    new_terrain[ny, nx] = 6
                    new_water_depth[ny, nx] = 0.2

def update_grid(terrain, water_depth, flow_speed):
    new_terrain = terrain.copy()
    new_water_depth = water_depth.copy()

    for y in range(terrain.shape[0]):
        for x in range(terrain.shape[1]):
            update_cell(x, y, terrain, water_depth, new_terrain, new_water_depth, flow_speed)

    return new_terrain, new_water_depth


class FloodSimulation:
    def __init__(self, terrain, water_depth, satellite_image):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Flood Simulation")
        self.clock = pygame.time.Clock()

        self.terrain = terrain
        self.water_depth = water_depth
        self.satellite_image = satellite_image

        scale_x = WINDOW_WIDTH / terrain.shape[1]
        scale_y = (WINDOW_HEIGHT - SLIDER_HEIGHT - 2 * SLIDER_MARGIN) / terrain.shape[0]
        self.SCALE = min(scale_x, scale_y)

        self.simulation_surface = pygame.Surface((terrain.shape[1] * CELL_SIZE, terrain.shape[0] * CELL_SIZE))
        self.satellite_surface = None
        if satellite_image is not None:
            self.satellite_surface = pygame.Surface((terrain.shape[1] * CELL_SIZE, terrain.shape[0] * CELL_SIZE))
            pygame.surfarray.blit_array(self.satellite_surface, self.satellite_image)

        self.ui_overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        self.opacity = 128
        self.show_satellite = self.satellite_surface is not None
        self.paused = True
        self.flow_speed = 1.0

        margin = 20
        self.opacity_slider_rect = pygame.Rect(WINDOW_WIDTH - 170, margin, 150, 10)
        self.flow_speed_slider_rect = pygame.Rect(WINDOW_WIDTH - 170, margin + 50, 150, 10)

        self.dragging_opacity = False
        self.dragging_flow_speed = False

        self.tools = ['dam', 'water', 'terrain', 'eraser']
        self.current_tool = 'dam'
        self.tool_sizes = {'dam': 10, 'water': 15, 'terrain': 20, 'eraser': 10}
        self.current_size = self.tool_sizes[self.current_tool]

    def get_grid_coordinates(self, mouse_pos):
        map_x = (WINDOW_WIDTH - self.simulation_surface.get_width() * self.SCALE) // 2
        map_y = (WINDOW_HEIGHT - self.simulation_surface.get_height() * self.SCALE) // 2
        grid_x = int((mouse_pos[0] - map_x) / self.SCALE)
        grid_y = int((mouse_pos[1] - map_y) / self.SCALE)
        return grid_x, grid_y

    def is_valid_grid_position(self, x, y):
        return 0 <= x < self.terrain.shape[1] and 0 <= y < self.terrain.shape[0]

    def draw_tools_panel(self):
        # Draw tools panel background
        tools_background = pygame.Rect(10, 150, 200, 200)
        pygame.draw.rect(self.ui_overlay, (0, 0, 0, 128), tools_background)

        font = pygame.font.Font(None, 24)
        y_offset = 160

        # Draw tool buttons
        for i, tool in enumerate(self.tools):
            tool_rect = pygame.Rect(20, y_offset + i * 30, 180, 25)
            color = (100, 100, 200) if tool == self.current_tool else (70, 70, 70)
            pygame.draw.rect(self.ui_overlay, color, tool_rect)

            text = font.render(f"{tool.capitalize()} (size: {self.tool_sizes[tool]})", True, WHITE)
            self.ui_overlay.blit(text, (25, y_offset + i * 30 + 5))

    def handle_tool_selection(self, pos):
        for i, tool in enumerate(self.tools):
            tool_rect = pygame.Rect(20, 160 + i * 30, 180, 25)
            if tool_rect.collidepoint(pos):
                self.current_tool = tool
                self.current_size = self.tool_sizes[tool]
                return True
        return False

    def apply_tool(self, x, y, size, terrain_value, water_value=0):
        """Apply tool effect to a circular area"""
        half_size = size // 2
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                ny, nx = y + dy, x + dx
                if self.is_valid_grid_position(nx, ny):
                    if (dx * dx + dy * dy) <= (half_size * half_size):
                        if terrain_value is not None:
                            self.terrain[ny, nx] = terrain_value
                        self.water_depth[ny, nx] = water_value

    def use_current_tool(self, x, y):
        if not self.is_valid_grid_position(x, y):
            return

        tool_effects = {
            'dam': (5, 0),  # terrain_value, water_value
            'water': (3, 1.0),
            'terrain': (4, 0),
            'eraser': (0, 0)
        }

        terrain_value, water_value = tool_effects[self.current_tool]
        self.apply_tool(x, y, self.current_size, terrain_value, water_value)

    def handle_mouse_wheel(self, y):
        self.tool_sizes[self.current_tool] = max(1, min(50, self.tool_sizes[self.current_tool] + y))
        self.current_size = self.tool_sizes[self.current_tool]

    def handle_slider(self, pos, slider_rect, min_value, max_value):
        """Generic slider handler"""
        if not slider_rect.collidepoint(pos):
            return None

        relative_x = min(max(0, pos[0] - slider_rect.x), slider_rect.width)
        return min_value + (relative_x / slider_rect.width) * (max_value - min_value)

    def handle_opacity_slider(self, pos):
        value = self.handle_slider(pos, self.opacity_slider_rect, 0, 255)
        if value is not None:
            self.opacity = value
            return True
        return False

    def handle_flow_speed_slider(self, pos):
        value = self.handle_slider(pos, self.flow_speed_slider_rect, 0.1, 3.0)
        if value is not None:
            self.flow_speed = value
            return True
        return False

    def draw_sliders(self):
        # Draw opacity slider
        pygame.draw.rect(self.ui_overlay, (0, 0, 0, 128),
                         (self.opacity_slider_rect.x - 10, self.opacity_slider_rect.y - 25, 170, 80))
        pygame.draw.rect(self.ui_overlay, SLIDER_COLOR, self.opacity_slider_rect)

        # Draw flow speed slider
        pygame.draw.rect(self.ui_overlay, SLIDER_COLOR, self.flow_speed_slider_rect)

        # Draw slider handles and labels
        font = pygame.font.Font(None, 24)

        # Opacity slider
        handle_pos = self.opacity_slider_rect.x + (self.opacity / 255) * self.opacity_slider_rect.width
        pygame.draw.rect(self.ui_overlay, SLIDER_HANDLE_COLOR,
                         (handle_pos - 5, self.opacity_slider_rect.y - 5, 10, 20))
        text = font.render(f"Opacity: {int(self.opacity / 2.55)}%", True, WHITE)
        self.ui_overlay.blit(text, (self.opacity_slider_rect.x, self.opacity_slider_rect.y - 25))

        # Flow speed slider
        handle_pos = self.flow_speed_slider_rect.x + ((self.flow_speed - 0.1) / 2.9) * self.flow_speed_slider_rect.width
        pygame.draw.rect(self.ui_overlay, SLIDER_HANDLE_COLOR,
                         (handle_pos - 5, self.flow_speed_slider_rect.y - 5, 10, 20))
        text = font.render(f"Flow Speed: {self.flow_speed:.1f}x", True, WHITE)
        self.ui_overlay.blit(text, (self.flow_speed_slider_rect.x, self.flow_speed_slider_rect.y - 25))

    def draw_simulation(self):
        self.simulation_surface.fill(WHITE)

        # Draw terrain and water
        for y in range(self.terrain.shape[0]):
            for x in range(self.terrain.shape[1]):
                color = self.get_cell_color(x, y)
                pygame.draw.rect(self.simulation_surface, color,
                                 (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Scale and center the simulation surface
        scaled_surface = pygame.transform.scale(
            self.simulation_surface,
            (int(self.simulation_surface.get_width() * self.SCALE),
             int(self.simulation_surface.get_height() * self.SCALE))
        )

        map_x = (WINDOW_WIDTH - scaled_surface.get_width()) // 2
        map_y = (WINDOW_HEIGHT - scaled_surface.get_height()) // 2

        self.screen.blit(scaled_surface, (map_x, map_y))

        # Draw satellite overlay if enabled
        if self.show_satellite and self.satellite_surface is not None:
            scaled_satellite = pygame.transform.scale(
                self.satellite_surface,
                (int(self.satellite_surface.get_width() * self.SCALE),
                 int(self.satellite_surface.get_height() * self.SCALE))
            )
            scaled_satellite.set_alpha(self.opacity)
            self.screen.blit(scaled_satellite, (map_x, map_y))

    def get_cell_color(self, x, y):
        terrain_type = self.terrain[y, x]
        water_depth = self.water_depth[y, x]

        if terrain_type == 5:  # Dam
            return BLACK
        elif terrain_type == 4:  # Terrain
            return BROWN
        elif terrain_type == 6:  # Flooded terrain
            water_color = get_water_color(water_depth)
            terrain_color = get_flooded_terrain_color()
            return tuple((w + t) // 2 for w, t in zip(water_color, terrain_color))
        elif terrain_type == 3:  # Water
            return get_water_color(water_depth)
        return WHITE  # Empty space

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.dragging_opacity = False
                    self.dragging_flow_speed = False
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4 or event.button == 5:  # Mouse wheel
                        self.handle_mouse_wheel(1 if event.button == 4 else -1)
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)

            if not self.paused:
                self.terrain, self.water_depth = update_grid(self.terrain, self.water_depth, self.flow_speed)

            self.draw_frame()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def handle_mouse_down(self, event):
        if event.button == 1:  # Left click
            if self.handle_opacity_slider(event.pos):
                self.dragging_opacity = True
            elif self.handle_flow_speed_slider(event.pos):
                self.dragging_flow_speed = True
            elif self.handle_tool_selection(event.pos):
                pass
        elif event.button == 3:  # Right click
            grid_x, grid_y = self.get_grid_coordinates(event.pos)
            self.use_current_tool(grid_x, grid_y)

    def handle_mouse_motion(self, event):
        if self.dragging_opacity:
            self.handle_opacity_slider(event.pos)
        elif self.dragging_flow_speed:
            self.handle_flow_speed_slider(event.pos)

    def handle_keydown(self, event):
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_TAB:
            self.show_satellite = not self.show_satellite

    def draw_frame(self):
        self.screen.fill(CHART_BG)
        self.draw_simulation()

        self.ui_overlay.fill((0, 0, 0, 0))
        self.draw_sliders()
        self.draw_tools_panel()

        if self.paused:
            font = pygame.font.Font(None, 36)
            pause_text = font.render("PAUSED", True, WHITE)
            text_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
            pygame.draw.rect(self.ui_overlay, (0, 0, 0, 128),
                             (text_rect.x - 10, text_rect.y - 5, text_rect.width + 20, text_rect.height + 10))
            self.ui_overlay.blit(pause_text, text_rect)

        self.screen.blit(self.ui_overlay, (0, 0))
def main():
    terrain, water_depth, _ = load_map("idk2.png")
    _, _, satellite_image = load_map("idk2.png", "idk.png")

    simulation = FloodSimulation(terrain, water_depth, satellite_image)
    simulation.run()

if __name__ == "__main__":
    main()