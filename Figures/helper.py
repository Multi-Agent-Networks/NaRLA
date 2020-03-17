def make_gif():
    X, Y, C = [],[],[]
    l = np.linspace(0,1,100)
    plt.plot(l,l,'--',linewidth=2)

    for i in range(200):
        X.append(np.random.uniform())
        Y.append(np.random.uniform())

        #   (x2 - x1) * (datay  - y1) - (y2 - y1) * (datax  - x1)
        c = (1. - 0.) * (Y[-1] - 0.) - (1. - 0.) * (X[-1] - 0.)
        C.append(c > 0)

        plt.scatter(X,Y,c=C,cmap='jet'); plt.title('Binary Classification Task'); 
        plt.xlim(0,1); plt.ylim(0,1)
        plt.savefig('Images/%05d.png' % i)


def figure():
    import matplotlib.pyplot as plt
    sep = .2
    for i in range(5):
        circle1 = plt.Circle((1, (0.3 + sep) * i), .1, color='black', fill=False)
        plt.gcf().gca().add_artist(circle1)

    circle1 = plt.Circle((1.5, 1), .1, color='black', fill=False)
    plt.gcf().gca().add_artist(circle1)
    plt.ylim(-.5, 2.5);
    plt.xlim(0.5, 2);
    plt.show()