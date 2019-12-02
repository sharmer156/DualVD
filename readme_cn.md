---
title: 2019-12-2 用于视觉对话中深度视觉理解的自适应双向编码模型—DualVD, 中科院信工所于静等
tags: 新建,模板,小书匠
grammar_cjkRuby: true
---


## AAAI2020论文】用于视觉对话中深度视觉理解的自适应双向编码模型—DualVD, 中科院信工所于静等

原创： 于静等人 [专知](javascript:void(0);) _1周前_ 导读近年来，**结合视觉和语言的跨媒体**人工智能技术取得了很大进展。其中，视觉对话任务要求模型同时具备推理、定位、语言表述等能力，对跨媒体智能提出了更大挑战。本文介绍了中科院信工所于静等AAAI2020的论文《DualVD: An Adaptive Dual Encoding Model for Deep Visual Understanding in Visual Dialogue》（AAAI 2020）, 该文针对视觉对话中涉及的图像内容范围广、多视角理解困难的问题，提出**一种用于刻画图像视觉和语义信息的自适应双向编码模型**——DualVD，从视觉目标、视觉关系、高层语义等多层面信息中自适应捕获回答问题的依据，同时通过可视化结果揭示不同信息源对于回答问题的贡献，具有较强的可解释性。该论文是和阿德莱德大学、北京航空航天大学、微软亚洲研究院共同完成。

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTyxngCFoOj67vVebtLWyC0icuL5glHDlp5U6fV9dVfCianbh5IhIH0F1A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**代码链接：****https://github.com/JXZe/DualVD**

论文链接：https://arxiv.org/abs/1911.07251

https://www.zhuanzhi.ai/paper/7cc3f5370a3c66f24e557fdf412c4f79

**动机**

综合分析语言和视觉等不同模态数据，对现实世界中的知识实现更加泛化的分析和推理，对推动人工智能的发展具有重要意义。近年来，跨媒体分析取得了显著进展，包括**跨媒体检索、视觉问答、指代理解、图像描述生成、视觉对话**等。其中，视觉对话任务要求模型根据图像、图像描述、对话历史回答当前问题。相比其他视觉-语言任务中主要关注特定的视觉目标或区域，视觉对话需要模型根据对话的推进，不断调整视角，关注问题涉及的多样的视觉信息。比如回答图1中的问题“Q1: Is the man on the skateboard?”，需要模型关注“the man”、“the skateboard”等前景信息，而问题“Q5: Is there sky in the picture?”又将视角转移到背景信息“sky”。除了Q1和Q5这类涉及表象层面（appearance-level）的问题，问题“Q4: Is he young or older?”又需要推理视觉内容得到高层语义信息。因此，**如何在对话过程中自适应地捕获回答问题所需的视觉线索是视觉对话中的重要挑战之一**。

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTMFtKF0sibWxJGklYWsYpSMteUSBJenveiaLYJESKnOibibfFNEHZ3JZqTg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**图1** DualVD模型基本思想。（左）模型输入；（右）视觉和语义信息理解模块。模型根据多模态线索推理当前问题的答案。

**核心思想**

认知学中双向编码理论（Dual-Coding Theory）认为：**人类大脑编码信息包括两种方式，即_视觉表象_和_关联文本. _**当被问到某个概念时，大脑会检索相关的视觉信息、语言信息或综合考虑上述两种信息。这种双向编码方式能够增强大脑的记忆和理解能力。作者受该理论启发，首先**提出了一种从视觉和语义两方面刻画图像信息的新框架**：视觉模块刻画图像中的主体目标和目标间的视觉关系，语义模块刻画图像中抽象的局部和全局高层语义信息。基于上述框架，作者提出了**一种自适应视觉信息选择模型DualVD (Dual Encoding Visual Dialogue)**：（1）模态内信息选择：由问题驱动，分别在视觉模块和语义模块中获得独立线索；（2）模态间信息选择：由问题驱动，获得视觉-语义的联合线索。

**论文的主要贡献有三点**：

*   **提出一种刻画图像信息的新框架，涵盖视觉对话中广泛的视觉内容；**

*   **提出一种自适应视觉信息选择模型，并支持显示地解释信息选择过程；**

*   **多个数据集上的实验结果显示，该模型优于大部分现有工作。**

**模型设计**

**视觉对话任务定义**：给定图像_**I**_，图像描述_**C**_和t-1轮的对话历史Ht=**{****_C_**,(**Q1,A1),...,(Q****t-1****,A****t-1****)**}, 以及当前轮问题_**Q**_，该任务要求从100个候选答案**A=(A1,A2,...,A100)** 中选择最佳答案。

DualVD模型结构如图2所示，模型核心结构分为两部分：**Visual-Semantic Dual Encoding**和**Adaptive Visual-Semantic Knowledge Selection**。

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFThBMTAxXraMaeqIHxSUWRYJ3JUxqW3UwFIQdVWs5pIa8d1ibs9ic14G7g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**图2** DualVD模型结构图

**1.  Visual-Semantic Dual Encodings**：提出刻画图像的视觉信息和语义信息的新框架，其中视觉信息采用场景图表示，语义信息采用多层面语义描述表示。

*   **Scene Graph Construction**：将每幅图像表示为一个场景图，同时刻画图像的目标和视觉关系信息。采用Faster-RCNN提取图像中N个目标区域，构成场景图上的结点，结点**i**的特征定义为hi；采用Zhang等提出的视觉关系编码器在GQA数据集上预训练，将给定图像中任何两个目标区域间的视觉关系编码为关系向量，构成场景图上的边，结点i和结点j间的关系向量定义为rij。相比现有方法采用关系类别表示场景图的边，作者考虑了视觉关系的多样性、歧义性，采用关系的嵌入表示能够更准确表达目标间的视觉关系。

*   **Multi-level Image Captions：**将每幅图像表示为多层面的语义描述，同时刻画图像的局部和全局语义信息。相比视觉特征，语言描述的优势在于能够更直接的为问题提供线索，避免了不同模态数据间的“异构鸿沟”。作者采用数据集提供的图像描述作为图像的全局语义信息；采用Feifei Li等提出的DenseCap提取描述细节的_k_条dense captions作为图像的局部语义信息。对全局和局部信息分别采用不同的LSTM提取特征，分别表示为C~和Z~={z1,z2,...,zk}。

**2. Adaptive Visual-Semantic Knowledge Selection**：基于上述图像的视觉和语义表示，作者提出一种问题自适应的信息选择模型—DualVD。基于问题的引导，DualVD的信息选择过程分两步：首先，模态内信息选择分别通过视觉模块（Visual Module）和语义模块（Semantic Module）提取视觉和语义信息；之后，模态间特征选择通过选择性视觉-语义融合（Selective visual-semantic fusion）汇聚视觉模块和语义模块中问题相关的线索。

*   ******Visual Module******

**(1) uestion-guided relation attention: **基于问题引导，捕获和问题最相关的视觉关系。首先，通过门控机制从对话历史中选择问题相关的信息更新问题表示：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTBCNkXsTf9BDc74UD0AsziafMxhgJzphWVSyr36TNRBlIumyCf1Q6icgw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

                基于问题新表示Qtg的引导，计算场景图中每个关系的注意力：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTxCqgUnhayHcIKWczgaVR3qn5S2B8rQtkfjxBwFQ6En0Dz4DuOUAHLw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

                基于注意力aij,更新场景图中每个关系的嵌入表示：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFT0icWq6iadDWoGI7XVM0VtMIFtaDHgT60wzmTjUAvZhoLA4sQGPqAamGw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**(2) Question-guided graph convolution:** 模块首先采用基于关系的图注意力网络（Relation-based GAT）获得感知视觉关系的目标表示。首先，对于场景图中的结点![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTSc82bXPyIhyQU3X0Ry4rCeN4QkhkbzkII4lwOW7mibVzYyQ6X41ibkqw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，计算该借点对邻居结点j在关系rij条件下的注意力：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTaO4oMWJBWnBOBnMgvOq0cgDdsriaEV5haf2CYSxgt0eVo9otiabiaPn2g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

            基于注意力βij更新场景图中每个结点的特征表示：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTlib7PUs1hFafVicJxPYDMQxS924EvwSlC0ibtuibwVAic0wF5Xr0S5q3Myg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**        (3) Object-relation Information fusion:** 采用门控机制融合感知关系的结点表示和原始结点表示：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTTBP4XBt7H94JkOqW5DaAHibSKLNJro9Kd64ACHAibNfFT5le6ZTKichug/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

为增强原始目标区域为回答问题提供的线索，作者基于原始目标区域的注意力分     布，融合目标区域表示得到增强的图像表示**_I_**：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTiaKntxeEZSPWgzUYQwXenu35UnlicP3eXAvKYHoAoqXGQmyLBib2VLcog/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTkBIgYUH9Wup2QtzpAuuDh6vYicZxdqLC1BQnbF5tgy2wkFYq6JYtQHQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

****  **Semantic Module **

**    (1) Question-guided semantic attention: ** 基于问题引导，对全局和局部语义描述mi∈{C,z1,z2,...,zk}计算注意力分布：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFT2Ex0f1cSx4K1iaAgF29EolOHsJLKAlC3TrRjIlrPPia2hAZm9gFFibfAA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

基于注意力**δ**iq分别更新全局和局部语义表示：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTEXxo34aQTW78JBBxFgcpzCCbZahNbyTice4jNsPdKCe6DT7vicmTibaQw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**    (2) Global-local information fusion:** 采用门控机制融合全局语义表示和局   部语义表示：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTbqiaCicWP2hHrYVg2er5lw2szbeYco92UdiaE5BOqK5IzMY3g1okcfKNQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**    (3) Selective Visual-Semantic Fusion:** 当被提问时，模型能够检索相关的视觉信息、语言信息或综合考虑上述两种信息。作者采用门控机制控制两种信息源对于回答问题的贡献，并获得最终的图像表示![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTSc82bXPyIhyQU3X0Ry4rCeN4QkhkbzkII4lwOW7mibVzYyQ6X41ibkqw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)：

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw0icRzeBBPFNj9VHWo91QyFTeyalYTo1V8fz86nJLKLzDomncvKR6a4ibRPicXVv6hhBfNyiceW30ummg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**3. Late Fusion Discriminative Decoder: **DualVD采用Late Fusion encoder和Discriminative decoder.模型在解码过程中，首先融合（拼接）更新后的问题表示、历史表示和图像表示，通过softmax得到在100个候选答案上的分布，排序选择最优的预测结果。作者表示，该模型和现有针对对话历史的研究工作具有互补优势，可以应用于更加复杂的encoder,如memory network, co-attention, adversarial network等。本篇论文重点证明所提出的视觉建模方法的有效性，因此采用了简单的Late Fusion encoder。

**实现结果**

作者在VisDial v0.9和VisDial v1.0上对模型的效果进行了验证。

**State-of-the-art comparison**

与现有算法相比，DualVD的结果超过现有大多数模型，略低于采用了多步推理和复杂attention机制的模型。

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw3mvCEeyvpMpJmr54S34U8xPC4Rib58MU1cYmUCIoLtE0uTRtuypiaGaqTPYiaH4u5nH4taby9oDJtsA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw3mvCEeyvpMpJmr54S34U8xfwDE49nKAicF9ypibKQxBXyfvZOqtiakY5NZW5EVDKLlJ63x32qIjqM8A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**Ablation Study**

作者做了充分的消融实验，验证模型各关键模块的作用，包括：ObjRep（目标特征）、RelRep(关系特征)、VisNoRel(视觉模块去掉关系嵌入表示)、VisMod(完整视觉模块)、GlCap(全局语义)、LoCap(局部语义)、SemMod(语义模块)、w/o ElMo (不用预训练语言模型)、DualVD(完整模型)。
实验结果表明，模型中的目标特征、关系特征、局部语义、全局语义对于提升回答问题的效果都起到了不同程度的作用。值得一提的是，相比传统图注意力模型，采用视觉关系的嵌入表示后，模型效果又有了进一步提升。

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw3mvCEeyvpMpJmr54S34U8xhKWMEJCEcqA09MxCxwrYNyACvbLnX3K0Cd7tNbO5gP6PalzteApkjg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&retryload=1)

**Interpretability**

该模型的一个优势是具有较强的可解释性，通过对attention weight、gate value的可视化，能够显示分析模型特征选择的过程。部分可视化结果如图3所示，。作者通过大量可视化结果分析得出以下观察：

   **视觉信息和语义信息对于回答问题的贡献取决于问题的复杂性和信息源的相关性。**涉及到目标表象信息的问题，模型会从视觉信息获得更多线索，例如图3中第一个个例子；当问题涉及到更加复杂的关系推理，或者语义信息包含了直接线索时，模型会倾向于从依赖语义信息中寻求答案，例如图3中的第二个例子。

   **视觉信息将为回答问题提供更重要的依据。**作者观察了所有测试数据后发现，来自视觉模块的累积gate value总是高于来自语义模块的累积gate value,表明在视觉对话任务中图像信息仍是回答问题的关键，对图像信息更准确、更全面的理解对于提升模型的对话能力十分重要。

   **模型能够根据问题的变化，自适应调整关注的信息。**在多轮对话中，如图3中的第三个例子，随着对话的推进，问题涉及前景、背景、语义关系等广泛的视觉内容，DualVD都能够有效捕捉到关键线索。

![](https://mmbiz.qpic.cn/mmbiz_png/AefvpgiaIPw3mvCEeyvpMpJmr54S34U8xXj3GXUvMXmmmHxJIDYibsnCialrRq1lxw3TDGKfCzgz0t7ibQ6ibuD02vQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

原文链接：

https://arxiv.org/pdf/1911.07251.pdf

**代码链接**：

https://github.com/JXZe/DualVD

**更多AAAI2020论文请关注登录专知网站www.zhuanzhi.ai, 实时更新关注：**

https://www.zhuanzhi.ai/topic/2001676361262408/vip

