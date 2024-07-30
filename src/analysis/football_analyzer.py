import numpy as np
from collections import defaultdict

class FootballAnalyzer:
    def __init__(self, pass_distance_threshold=20):
        self.pass_distance_threshold = pass_distance_threshold
        
        self.passes = {"Spain": 0, "Portugal": 0}
        self.total_passes = 0
        self.ball_distance_covered = 0
        self.player_distance_covered = defaultdict(float)
        
        self.last_ball_pos = None
        self.last_player_positions = {}
        self.last_team_with_ball = None
        
        self.stats = {
            'possession': {},
            'passes': {"Spain": 0, "Portugal": 0},
            'ball_distance': 0,
            'player_distance': {}
        }

    def update(self, frame_number, ball_position, player_positions, team_colors):
        self.update_ball_stats(ball_position)
        self.update_player_stats(player_positions)
        self.update_passes_and_possession(ball_position, player_positions, team_colors)
        
        self.last_ball_pos = ball_position
        self.last_player_positions = player_positions
        
        self.compile_stats()

    def update_ball_stats(self, ball_position):
        if self.last_ball_pos is not None:
            self.ball_distance_covered += np.linalg.norm(np.array(ball_position) - np.array(self.last_ball_pos))

    def update_player_stats(self, player_positions):
        for player_id, position in player_positions.items():
            if player_id in self.last_player_positions:
                distance = np.linalg.norm(np.array(position) - np.array(self.last_player_positions[player_id]))
                self.player_distance_covered[player_id] += distance

    def update_passes_and_possession(self, ball_position, player_positions, team_colors):
        if self.last_ball_pos is None:
            return
        
        ball_movement = np.linalg.norm(np.array(ball_position) - np.array(self.last_ball_pos))
        
        if ball_movement > self.pass_distance_threshold:
            closest_player = min(player_positions.items(), key=lambda x: np.linalg.norm(np.array(x[1]) - np.array(ball_position)))
            closest_player_id, _ = closest_player
            current_team_color = team_colors[closest_player_id]
            
            current_team = "Spain" if current_team_color == (255, 255, 255) else "Portugal"
            
            self.passes[current_team] += 1
            self.total_passes += 1
            self.last_team_with_ball = current_team

    def compile_stats(self):
        total_passes = sum(self.passes.values())
        self.stats.update({
            'possession': {team: passes / total_passes if total_passes > 0 else 0 
                        for team, passes in self.passes.items()},
            'passes': dict(self.passes),  # Make sure to update the passes
            'ball_distance': self.ball_distance_covered,
            'player_distance': dict(self.player_distance_covered)
        })

    def get_stats(self):
        return self.stats