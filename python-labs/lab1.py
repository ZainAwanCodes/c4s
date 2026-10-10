student="Ali"
quiz =18
assignment = 24
final = 45
total_marks= quiz+ assignment+ final
print(f"Student Name : {student} Total Marks : {total_marks}")
percentange = (total_marks / 100) * 100
print("Percentage:",percentange,"%")

if (total_marks>50):
    print("Pass")
else:
    print("Fail")