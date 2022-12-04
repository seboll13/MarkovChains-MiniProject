def single_queen_conflict_calculator(queen_id) -> int:
    """ This function calculates the number of conflicts for a single queen."""
    conflicts = 0
    for i in range(4):
        if i != queen_id:
            if abs(q_coordinates[queen_id][0] - q_coordinates[i][0]) == abs(
                    q_coordinates[queen_id][1] - q_coordinates[i][1]):
                conflicts += 1
    return conflicts

q_coordinates = [(1,0),(2,1),(1,3),(3,3)]

q1_conflicts = single_queen_conflict_calculator(0)
print(q1_conflicts)


def single_queen_conflitto_calculator(queen_id) -> int:
    conflittos = 0
    for i in range(4):
        if i != queen_id:
            # diagonal calculator
            if abs(q_coordinates[queen_id][0] - q_coordinates[i][0]) == abs(
                    q_coordinates[queen_id][1] - q_coordinates[i][1]):
                conflittos += 1
            # same row
            elif q_coordinates[queen_id][0] == q_coordinates[i][0]:
                conflittos += 1
            # same column
            elif q_coordinates[queen_id][1] == q_coordinates[i][1]:
                conflittos += 1
    return conflittos

q1_conflicts = single_queen_conflitto_calculator(0)
print(q1_conflicts)