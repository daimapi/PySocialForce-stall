import numpy as np

num = np.array([[1], [1], [1], [1], [1], [1]])
arrivedf_mask = np.array([True, True, False, True, False, False])
a = 1  # You need to define the value of 'a'

print(np.where(arrivedf_mask)) # (array([0, 1, 3], dtype=int32),)
# Check if 'a' is a valid index

    # Update the value in the original array
num[np.where(arrivedf_mask)[0][a]] += 1

print(num)

print("---------------------------------------------------------")


force = np.array([[],[],[],[],[],[],[],[],[],[]])
m = np.array([False, False, False, False, False, False])

ans = np.zeros((m.shape[0],force.shape[0]))
print(force)
print(type(force))
print(force.shape)
force = np.rot90(force,3)
for a in range(m.shape[0]):
    if m[a]:
        ans[a] = force[0]
        force = np.delete(force, 0, 0)
print(force)
print(type(force))
print(force.shape)
force = list(np.rot90(ans))
abc = sum(force)
print(abc)
#######################################
print("##############################################")
#######################################
force = np.array([[1,11],[2,22],[3,33],[4,44]])
m = np.array([False, True, True, True, False, True])

ans = np.zeros((m.shape[0],2))
for a in range(m.shape[0]):
    if m[a]:
        ans[a] = force[0]
        force = np.delete(force, 0, 0)
print(ans)
#c=np.array([1,2])
#a=3
#b=2
#print(a>b)
#print([a>b])
#print([a>b,a>b])
