# lunar_ui.py
import math
import random
import numpy as np
import pygame
import gymnasium as gym

SCALE = 30.0
VIEWPORT_W = 600
VIEWPORT_H = 400


class UpgradedLunarUI(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.ep_score = 0.0
        self._font = None

        # Pre-generate fixed background stars so they don't flicker randomly
        rng = random.Random(7)
        self._stars = [
            (
                rng.randint(0, VIEWPORT_W),
                rng.randint(0, int(VIEWPORT_H * 0.82)),
                rng.choice([1, 1, 1, 2]),
                rng.randint(130, 245),
            )
            for _ in range(95)
        ]

        # Override the unwrapped environment's render() so env.step() in "human"
        # mode and RecordVideo in "rgb_array" mode both use the upgraded visuals
        self.env.unwrapped.render = self._custom_render

    def reset(self, **kwargs):
        self.ep_score = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.ep_score += float(reward)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._custom_render()

    @staticmethod
    def _w2s(wx, wy):
        """Convert Box2D world coordinates to 600x400 screen coordinates."""
        return int(wx * SCALE), int(VIEWPORT_H - wy * SCALE)

    def _custom_render(self):
        uw = self.env.unwrapped
        if uw.render_mode is None:
            return None

        if uw.screen is None:
            pygame.init()
            pygame.font.init()
            if uw.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("LunarLander-v3")
                uw.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            else:
                uw.screen = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        if uw.clock is None:
            uw.clock = pygame.time.Clock()
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", 14, bold=True)

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        # 1. Deep Space Sky & Distant Earth
        surf.fill((10, 13, 22))
        for sx, sy, r, b in self._stars:
            pygame.draw.circle(surf, (b, b, min(255, b + 15)), (sx, sy), r)

        # Subtle distant Earth in top-right sky
        pygame.draw.circle(surf, (38, 78, 135), (515, 65), 22)
        pygame.draw.circle(surf, (65, 125, 195), (512, 62), 19)
        pygame.draw.circle(surf, (10, 13, 22), (522, 58), 18)  # Crescent shadow

        # 2. Lunar Terrain (Replaces the blinding white ground with shaded moon rock)
        terrain_ridge = []
        for i, poly in enumerate(uw.sky_polys):
            if i == 0:
                terrain_ridge.append(self._w2s(poly[0][0], poly[0][1]))
            terrain_ridge.append(self._w2s(poly[1][0], poly[1][1]))

        if len(terrain_ridge) >= 2:
            ground_poly = terrain_ridge + [(VIEWPORT_W, VIEWPORT_H), (0, VIEWPORT_H)]
            pygame.draw.polygon(surf, (46, 50, 58), ground_poly)

            # Sub-surface shadow layer
            lower_ridge = [(x, y + 8) for x, y in terrain_ridge]
            sub_poly = lower_ridge + [(VIEWPORT_W, VIEWPORT_H), (0, VIEWPORT_H)]
            pygame.draw.polygon(surf, (32, 35, 42), sub_poly)

            # Lit lunar dust top edge
            pygame.draw.lines(surf, (135, 142, 155), False, terrain_ridge, 3)

        # 3. Landing Pad & Upgraded Flags
        hx1, hy = self._w2s(uw.helipad_x1, uw.helipad_y)
        hx2, _ = self._w2s(uw.helipad_x2, uw.helipad_y)

        # Metallic pad deck
        pygame.draw.rect(surf, (75, 82, 95), (hx1, hy, hx2 - hx1, 6), border_radius=2)
        pygame.draw.line(surf, (220, 175, 45), (hx1 + 4, hy + 1), (hx2 - 4, hy + 1), 2)

        # Left & Right Flags
        for fx in (hx1, hx2):
            pole_top = hy - int(40 / SCALE * SCALE)
            pygame.draw.line(surf, (190, 200, 215), (fx, hy), (fx, pole_top), 2)
            flag_poly = [(fx, pole_top + 2), (fx, pole_top + 14), (fx + 18, pole_top + 8)]
            pygame.draw.polygon(surf, (235, 185, 35), flag_poly)
            pygame.draw.polygon(surf, (255, 225, 110), flag_poly, 1)
            pygame.draw.circle(surf, (0, 230, 140), (fx, pole_top), 3)

        # 4. Real Box2D Engine Exhaust Particles (Glowing flames instead of blocks)
        for obj in uw.particles:
            px, py = self._w2s(obj.position.x, obj.position.y)
            ttl = max(0.0, min(1.0, getattr(obj, "ttl", 0.5)))
            radius = max(2, int(6 * ttl))

            # Outer orange/red glow + hot white/yellow core
            outer_col = (235, int(110 * ttl + 40), 25)
            core_col = (255, 240, int(180 * ttl + 50))
            pygame.draw.circle(surf, outer_col, (px, py), radius + 2)
            pygame.draw.circle(surf, core_col, (px, py), max(1, radius - 1))

        # 5. Upgraded Spacecraft & Landing Legs (drawn on the exact Box2D bodies)
        if uw.lander is not None:
            self._draw_lander_and_legs(surf, uw)

        # 6. Minimal Corner Score Pill (Unobtrusive)
        score_txt = self._font.render(f"SCORE: {self.ep_score:+6.1f}", True, (215, 225, 240))
        surf.blit(score_txt, (12, 10))

        uw.screen.blit(surf, (0, 0))

        if uw.render_mode == "human":
            pygame.event.pump()
            uw.clock.tick(uw.metadata["render_fps"])
            pygame.display.flip()
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(uw.screen)), axes=(1, 0, 2)
            )

    def _draw_lander_and_legs(self, surf, uw):
        lander = uw.lander
        lx, ly = lander.position.x, lander.position.y
        angle = lander.angle
        ca, sa = math.cos(angle), math.sin(angle)

        def tf(px_local, py_local):
            """Transform local pixel offsets (matching LunarLander's 30px scale) to screen."""
            wx = lx + (px_local / SCALE) * ca - (py_local / SCALE) * sa
            wy = ly + (px_local / SCALE) * sa + (py_local / SCALE) * ca
            return self._w2s(wx, wy)

        # Draw Landing Legs using actual Box2D leg bodies + footpads
        for side, leg in zip((-1, 1), uw.legs):
            for f in leg.fixtures:
                trans = f.body.transform
                path = [self._w2s(*(trans * v)) for v in f.shape.vertices]
                leg_col = (70, 225, 135) if leg.ground_contact else (155, 165, 180)
                pygame.draw.polygon(surf, leg_col, path)
                pygame.draw.polygon(surf, (85, 95, 110), path, 1)

            # Secondary shock strut & footpad
            hip = tf(side * 12, -4)
            foot = self._w2s(leg.position.x, leg.position.y - 0.12)
            pad_col = (70, 225, 135) if leg.ground_contact else (190, 200, 215)
            pygame.draw.line(surf, (120, 130, 145), hip, foot, 2)
            pygame.draw.line(surf, pad_col, (foot[0] - 5, foot[1]), (foot[0] + 5, foot[1]), 3)

        # Dark Metallic Engine Nozzle Bell
        nozzle = [tf(-6, -9), tf(6, -9), tf(10, -15), tf(-10, -15)]
        pygame.draw.polygon(surf, (60, 68, 80), nozzle)
        pygame.draw.polygon(surf, (130, 140, 155), nozzle, 1)

        # Exact Box2D Lander Hull (Silver/White Upper Cabin + Gold Thermal Foil Base)
        hull_pts = [tf(-14, 17), tf(-17, 0), tf(-17, -10), tf(17, -10), tf(17, 0), tf(14, 17)]
        pygame.draw.polygon(surf, (220, 226, 235), hull_pts)

        # Gold Foil Descent Stage Band (bottom portion of the hull)
        foil_pts = [tf(-17, -1), tf(-17, -10), tf(17, -10), tf(17, -1)]
        pygame.draw.polygon(surf, (210, 155, 45), foil_pts)
        pygame.draw.line(surf, (245, 200, 90), tf(-17, -1), tf(17, -1), 1)

        # Crisp Hull Outline & Panel Seam
        pygame.draw.polygon(surf, (75, 85, 102), hull_pts, 2)

        # Cockpit Visor Window
        visor = [tf(-8, 11), tf(8, 11), tf(10, 4), tf(-10, 4)]
        pygame.draw.polygon(surf, (28, 52, 85), visor)
        pygame.draw.polygon(surf, (110, 175, 230), visor, 1)

        # Side RCS Thruster Nubs
        for side in (-1, 1):
            rcs = [tf(side * 15, 9), tf(side * 19, 9), tf(side * 19, 5), tf(side * 15, 5)]
            pygame.draw.polygon(surf, (90, 100, 115), rcs)