def status(mark):
    if mark>=50:
        print("Pass")
    else:
        print("Fail")
status(50)
def percentange(mark,total_mark):
    percentage = (mark/total_mark)*100
    return percentage
print(percentange(50,100))