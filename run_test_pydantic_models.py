from mosa.models import Group, Population

group1 = Group(
    name = "group1",
    data = (-10.0,5.0),
    )

print(group1)

group2 = Group(
    name = "group2",
    data = ["A", "B", "C", "D"],
    change_value_move = 1.0,
    insert_or_delete_move = 0.5,
    swap_move = 0.25
    )

print(group2)

pop1 = Population(
    groups = [group1,group2])

print(pop1)
