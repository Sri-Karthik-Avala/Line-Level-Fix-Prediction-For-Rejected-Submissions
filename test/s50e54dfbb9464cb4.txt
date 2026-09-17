class Node:
    def __init__(self, id):
        self.id = id
        self.parent = None
        self.left = None
        self.right = None

root = None
n = int(input())
node_list = []
for i in range(n):
    node_list.append(Node(i))

for i in range(n):
    id, left, right = list(map(int, input().split()))
    node = node_list[id]
    if left > -1:
        node_list[left].parent = node
        node.left = node_list[left]
    if right > -1:
        node_list[right].parent = node
        node.right = node_list[right]

root = None
for node in node_list:
    if node.parent is None:
        root = node
        break

def preorder(node):
    if node is None:
        return
    print(' ' + str(node.id), end='')
    preorder(node.left)
    preorder(node.right)

def inorder(node):
    if node is None:
        return
    
    inorder(node.left)
    print(' ' + str(node.id), end='')
    inorder(node.right)

def postorder(node):
    if node is None:
        return
    print(' ' + str(node.id), end='')
    postorder(node.left)
    postorder(node.right)

print('Preorder')
preorder(root)
print('')

print('Inorder')
inorder(root)
print('')

print('Postorder')
postorder(root)
print('')
