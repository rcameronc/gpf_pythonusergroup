def plot_from_samples(m, X, Y, parameters, parameter_samples, thin):

    f = plt.figure(figsize=(12, 6))
    a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
    a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
    a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])

    xx = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    Fpred, Ypred = [], []
    num_samples = len(parameter_samples[0])
    for i in range(0, num_samples, thin):
        for parameter, var_samples in zip(parameters, parameter_samples):
            parameter.assign(var_samples[i])
        Ypred.append(m.predict_y(xx)[0])
        Fpred.append(np.squeeze(m.predict_f_samples(xx, 1)))

    for i in range(m.likelihood.num_classes):
        x = X[Y.reshape(-1) == i]
        (points,) = a3.plot(x, x * 0, ".")
        color = points.get_color()
        for F in Fpred:
            a1.plot(xx, F[:, i], color=color, lw=0.2, alpha=1.0)
        for Yp in Ypred:
            a2.plot(xx, Yp[:, i], color=color, lw=0.5, alpha=1.0)

    a2.set_ylim(-0.1, 1.1)
    a2.set_yticks([0, 1])
    a2.set_xticks([])

    a3.set_xticks([])
    a3.set_yticks([])

    a3.set_title("inputs X")
    a2.set_title(
        "predicted mean label value \
                 $\mathbb{E}_{q(\mathbf{u})}[y^*|x^*, Z, \mathbf{u}]$"
    )
    a1.set_title(
        "posterior process samples \
                $\int d\mathbf{u} q(\mathbf{u})p(f^*|\mathbf{u}, Z, x^*)$"
    )
