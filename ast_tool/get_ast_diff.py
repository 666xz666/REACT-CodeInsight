def _get_diff(r, ty):
    # print(r)
    if ty == 0:
        global diff_list
        p = (0, 0)
        for c in r[1:]:
            if c[0] - p[0] == 1 and c[1] - p[1] == 0:
                diff_list[0].append(p[0])
            elif c[0] - p[0] == 0 and c[1] - p[1] == 1:
                diff_list[1].append(p[1])
            # if c[0] - p[0] == 1:
            # 	if c[1] - p[1] == 1:
            # 		print(l1[p[0]])
            # 		print(type(l1[p[0]]))
            # 	else:
            # 		print('-'+str(l1[p[0]]))
            # else:
            # 	print('+'+str(l2[p[1]]))
            p = c
    if ty == 1:
        global code_structure, old_list, new_list
        code_structure = ''
        p = (0, 0)
        for c in r[1:]:
            if c[0] - p[0] == 1:
                if c[1] - p[1] == 1:
                    code_structure += (old_list[p[0]] + ' ')
                else:
                    code_structure += '-' + old_list[p[0]] + ' '
            else:
                code_structure += '+' + new_list[p[1]] + ' '
            p = c


def get_diff(l1, l2, ty):
    n = len(l1)
    m = len(l2)

    v = [{0: [(0, 0)]}, {}]
    find = 0
    c1 = (0, 0)
    while c1[0] < n and c1[1] < m and str(l1[c1[0]]) == str(l2[c1[1]]):
        c1 = (c1[0] + 1, c1[1] + 1)
        v[0][0].append(c1)
    if c1 == (n, m):
        _get_diff(v[0][0], ty)
        find = 1

    pre = 1
    cur = 0

    for d in range(1, n + m + 1):
        if find == 1:
            break
        pre = (pre + 1) % 2
        cur = (cur + 1) % 2
        v[cur].clear()
        # print(d)
        for k in range(d, -d - 1, -2):  # k = x - y
            c = [(-1, -1)]
            t = k + 1
            if t < d:
                r = []
                for c2 in v[pre][t]:
                    r.append(c2)
                c1 = (r[-1][0], r[-1][1] + 1)
                r.append(c1)
                while c1[0] < n and c1[1] < m and str(l1[c1[0]]) == str(l2[c1[1]]):
                    # print(l1[c1[0]], l2[c1[1]])
                    c1 = (c1[0] + 1, c1[1] + 1)
                    r.append(c1)
                if c1 == (n, m):
                    find = 1
                    _get_diff(r, ty)
                    break
                c = r
            # print(k,c)
            t = k - 1
            if t > -d:
                r = []
                for c2 in v[pre][t]:
                    r.append(c2)
                c1 = (r[-1][0] + 1, r[-1][1])
                r.append(c1)
                while c1[0] < n and c1[1] < m and str(l1[c1[0]]) == str(l2[c1[1]]):
                    c1 = (c1[0] + 1, c1[1] + 1)
                    r.append(c1)
                if c1[0] > c[-1][0]:
                    c = r
                if c1 == (n, m):
                    find = 1
                    _get_diff(r, ty)
                    break
            # print(k,r)
            v[cur][k] = c
        # print(k,c)


def get_ast_diff(l1, l2):
    global diff_list, old_list, new_list
    diff_list = [[], []]
    old_list = l1
    new_list = l2

    get_diff(l1, l2, ty=0)
    get_diff(l1, l2, ty=1)

    return code_structure


if __name__ == '__main__':
    ###
    # 输入序列
    l1 = ["a", "b", "c", "d", 'e']
    l2 = ["a", "b", "e", "d", 'f']

    print(get_ast_diff(l1, l2))
