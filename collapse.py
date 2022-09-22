f=open("pred.de",encoding='utf-8').readlines()
o=open("pred.de.collapse","w",encoding='utf-8')
for line in f:
    words=line[:-1].split(' ')
    l = len(words)
    s = ''
    for i in range(l):
        if i == 0 or words[i] != words[i - 1]:
            if words[i] != '<blank>':
                s = s + words[i]
                if i != l - 1:
                    s = s + ' '
    s = s + '\n'
    o.write(s)
