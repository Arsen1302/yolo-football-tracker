from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance=70):
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        assigned_player = -1
        minimum_distance = self.max_player_ball_distance

        for player_id, player in players.items():
            player_bbox = player["bbox"]
            distance_to_left_foot = measure_distance(
                (player_bbox[0], player_bbox[3]),
                ball_position,
            )
            distance_to_right_foot = measure_distance(
                (player_bbox[2], player_bbox[3]),
                ball_position,
            )
            distance = min(distance_to_left_foot, distance_to_right_foot)

            if distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
