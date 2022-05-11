# CLIP-explainability

## Create a CLIP dream!

## How CLIP Dream works?

The CLIP model embeds text and images into the same embedding space. This embedding space is defined by $S^{511} \subset \mathbb{R}^{512}$.

Given an image $I \in \R^{3\cdot H \cdot W}$, we obtain its CLIP embedding $E = f(I) \in \mathbb{R}^{512}$, where $f: \R^{3\cdot H \cdot W} \rightarrow S^{511}$ is the CLIP image encoder (ViT-B/32). We then compute the Jacobian of $E$ with respect to $I$, $J = \frac{dE}{dI} \in M_{512 \times 3 \cdot H \cdot W}(\mathbb{R})$. This Jacobian resembles the local influence of the image pixels, $I$, on the image embedding, $E$. 

We compute the singular value decomposition of $J$, $J = U \cdot \text{diag}(S) \cdot V^T$, where $U \in M_{512 \times 512}(\mathbb{R})$, $S \in \mathbb{R}^{512}$, and $V^T \in M_{512 \times 3 \cdot H \cdot W}(\mathbb{R})$. The columns of $U$ are the ordered left singular vectors of $J$, which form an orthonomal basis of the tangent space of $E \in S^{511}$ and resemble the principal directions of the image of $J$. 

We choose one of these singular vectors, $V = U_{(i,\ \cdot)}$, to get an orthonormal vector to $E$ (alternatively, we may choose a random vector orthonormal to $E$). Using these vectors, we sample points along the surface of the embedding space in the direction of $V$ and starting from $E$: $p_k = \cos(k \theta) \cdot E + \sin(k \theta) \cdot V$. By optimizing the entries of $I$ to minimize $\Vert f(I) - p_i \Vert$, we find a sequence of images $ \{ I=I_0, \ I_1, ..., \ I_n \} \subset \mathbb{R}^{3\cdot H\cdot W}$ satisfying $f(I_i)\approx p_i$ for all $i \geq 1$. Turning this sequence of frames into a video gives the CLIP dream visualization!