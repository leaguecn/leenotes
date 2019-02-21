

**TOC**

<!-- MarkdownTOC levels="1,2,3,4" autolink="true" style="ordered" -->

1. [Explainer: What is a quantum computer?](#explainer-what-is-a-quantum-computer)
    1. [What is a qubit?](#what-is-a-qubit)
    1. [What is superposition?](#what-is-superposition)
    1. [What is entanglement?](#what-is-entanglement)
    1. [What is decoherence?](#what-is-decoherence)
    1. [What is quantum supremacy?](#what-is-quantum-supremacy)
    1. [Where is a quantum computer likely to be most useful first?](#where-is-a-quantum-computer-likely-to-be-most-useful-first)
1. [The real reason America is scared of Huawei: internet-connected everything](#the-real-reason-america-is-scared-of-huawei-internet-connected-everything)
    1. [WHAT IS 5G?](#what-is-5g)
    1. [WHY IS IT BETTER?](#why-is-it-better)
    1. [WHAT ARE THE SECURITY RISKS?](#what-are-the-security-risks)
    1. [CAN 5G BE MADE SECURE?](#can-5g-be-made-secure)
    1. [WHY IS HUAWEI’S 5G CAUSING SO MUCH CONCERN?](#why-is-huawei%E2%80%99s-5g-causing-so-much-concern)

<!-- /MarkdownTOC -->

-----



## [Explainer: What is a quantum computer?](https://www.technologyreview.com/s/612844/what-is-quantum-computing/)

![](https://github.com/leaguecn/leenotes/raw/master/img/What-is-a-quantum-computer.jpg)

**How it works, why it’s so powerful, and where it’s likely to be most useful first**

*by [Martin Giles](https://www.technologyreview.com/profile/martin-giles/) January 29, 2019*

Aquantum computer harnesses some of the almost-mystical phenomena of quantum mechanics to deliver huge leaps forward in processing power. Quantum machines promise to outstrip even the most capable of today’s—and tomorrow’s—supercomputers.

They won’t wipe out conventional computers, though. Using a classical machine will still be the easiest and most economical solution for tackling most problems. But quantum computers promise to power exciting advances in various fields, from materials science to pharmaceuticals research. Companies are already experimenting with them to develop things like lighter and more powerful batteries for electric cars, and to help create novel drugs.

The secret to a quantum computer’s power lies in its ability to generate and manipulate quantum bits, or qubits.

### What is a qubit?

![](https://cdn.technologyreview.com/i/images/quantumexplainer-01.png?sw=280&cx=0&cy=0&cw=1600&ch=1600)

Today's computers use bits—a stream of electrical or optical pulses representing *1*s or *0*s. Everything from your tweets and e-mails to your iTunes songs and YouTube videos are essentially long strings of these binary digits.

Quantum computers, on the other hand, use qubits, which are typically subatomic particles such as electrons or photons. Generating and managing qubits is a scientific and engineering challenge. Some companies, such as IBM, Google, and Rigetti Computing, use superconducting circuits cooled to temperatures colder than deep space. Others, like IonQ, trap individual atoms in electromagnetic fields on a silicon chip in ultra-high-vacuum chambers. In both cases, the goal is to isolate the qubits in a controlled quantum state.

Qubits have some quirky quantum properties that mean a connected group of them can provide way more processing power than the same number of binary bits. One of those properties is known as superposition and another is called entanglement.

### What is superposition?

![](https://cdn.technologyreview.com/i/images/quantumexplainer-02.png?sw=280&cx=0&cy=0&cw=1600&ch=1600)

Qubits can represent numerous possible combinations of *1*and *0*at the same time. This ability to simultaneously be in multiple states is called superposition. To put qubits into superposition, researchers manipulate them using precision lasers or microwave beams.

Thanks to this counterintuitive phenomenon, a quantum computer with several qubits in superposition can crunch through a vast number of potential outcomes simultaneously. The final result of a calculation emerges only once the qubits are measured, which immediately causes their quantum state to “collapse” to either *1*or *0*. []()

### What is entanglement?

![](https://cdn.technologyreview.com/i/images/quantumexplainer-04.png?sw=280&cx=0&cy=0&cw=1600&ch=1600)

Researchers can generate pairs of qubits that are “entangled,” which means the two members of a pair exist in a single quantum state. Changing the state of one of the qubits will instantaneously change the state of the other one in a predictable way. This happens even if they are separated by very long distances.

Nobody really knows quite how or why entanglement works. It even baffled Einstein, who famously described it as “spooky action at a distance.” But it’s key to the power of quantum computers. In a conventional computer, doubling the number of bits doubles its processing power. But thanks to entanglement, adding extra qubits to a quantum machine produces an exponential increase in its number-crunching ability.

Quantum computers harness entangled qubits in a kind of quantum daisy chain to work their magic. The machines’ ability to speed up calculations using specially designed quantum algorithms is why there’s so much buzz about their potential.

That’s the good news. The bad news is that quantum machines are way more error-prone than classical computers because of decoherence.

### What is decoherence?

![](https://cdn.technologyreview.com/i/images/quantumexplainer-03.png?sw=280&cx=0&cy=0&cw=1600&ch=1600)

The interaction of qubits with their environment in ways that cause their quantum behavior to decay and ultimately disappear is called decoherence. Their quantum state is extremely fragile. The slightest vibration or change in temperature—disturbances known as “noise” in quantum-speak—can cause them to tumble out of superposition before their job has been properly done. That’s why researchers do their best to protect qubits from the outside world in those supercooled fridges and vacuum chambers.

But despite their efforts, noise still causes lots of errors to creep into calculations. [Smart quantum algorithms](https://www.technologyreview.com/s/611139/the-worlds-first-quantum-software-superstore-or-so-it-hopes-is-here/) can compensate for some of these, and adding more qubits also helps. However, it will likely take thousands of standard qubits to create a single, highly reliable one, known as a “logical” qubit. This will sap a lot of a quantum computer’s computational capacity.

And there’s the rub: so far, researchers haven’t been able to generate more than 128 standard qubits (see our qubit counter [here](http://www.qubitcounter.com/)). So we’re still many years away from getting quantum computers that will be broadly useful.

That hasn’t dented pioneers’ hopes of being the first to demonstrate “quantum supremacy.”

### What is quantum supremacy?

![](https://cdn.technologyreview.com/i/images/quantumexplainer-05.png?sw=280&cx=0&cy=0&cw=1600&ch=1600)

It’s the point at which a quantum computer can complete a mathematical calculation that is demonstrably beyond the reach of even the most powerful supercomputer.

It’s still unclear exactly how many qubits will be needed to achieve this because researchers keep finding new algorithms to boost the performance of classical machines, and supercomputing hardware keeps getting better. But researchers and companies are working hard to claim the title, [running tests](https://www.technologyreview.com/s/612381/google-has-enlisted-nasa-to-help-it-prove-quantum-supremacy-within-months/) against some of the world’s most powerful supercomputers.

There’s plenty of debate in the research world about [just how significant achieving this milestone will be](https://www.technologyreview.com/s/610274/google-thinks-its-close-to-quantum-supremacy-heres-what-that-really-means/). Rather than wait for supremacy to be declared, companies are already starting to experiment with quantum computers made by companies like IBM, Rigetti, and D-Wave, a Canadian firm. Chinese firms like Alibaba are also offering access to quantum machines. Some businesses are buying quantum computers, while others are using ones made available [through cloud computing services](https://www.technologyreview.com/s/611962/faster-quantum-computing-in-the-cloud/).

### Where is a quantum computer likely to be most useful first?

![](https://cdn.technologyreview.com/i/images/quantumexplainer-06.png?sw=280&cx=0&cy=0&cw=1600&ch=1600)

One of the most promising applications of quantum computers is for [simulating the behavior of matter](https://www.technologyreview.com/s/603794/chemists-are-first-in-line-for-quantum-computings-benefits/) down to the molecular level. Auto manufacturers like Volkswagen and Daimler are using quantum computers to simulate the chemical composition of electrical-vehicle batteries to help find new ways to improve their performance. And pharmaceutical companies are leveraging them to analyze and compare compounds that could lead to the creation of new drugs.

The machines are also great for optimization problems because they can crunch through vast numbers of potential solutions extremely fast. Airbus, for instance, is using them to help calculate the most fuel-efficient ascent and descent paths for aircraft. And Volkswagen has unveiled a service that calculates the optimal routes for buses and taxis in cities in order to minimize congestion. Some researchers also think the machines could be used [to accelerate artificial intelligence](https://www.technologyreview.com/s/612435/machine-learning-meet-quantum-computing/).

It could take quite a few years for quantum computers to achieve their full potential. Universities and businesses working on them are facing [a shortage of skilled researchers](https://www.technologyreview.com/s/612071/us-takes-first-step-towards-creating-a-quantum-computing-workforce/) in the field—and [a lack of suppliers](https://www.technologyreview.com/s/612760/quantum-computers-component-shortage/) of some key components. But if these exotic new computing machines live up to their promise, they could transform entire industries and turbocharge global innovation.



--------------
2019-02-19

## [The real reason America is scared of Huawei: internet-connected everything](https://www.technologyreview.com/s/612874/the-real-reason-america-is-scared-of-huawei-internet-connected-everything/)

![](https://github.com/leaguecn/leenotes/raw/master/img/5gfinal1.jpg)


**Five things you need to know about 5G, the next generation of wireless tech that’s fueling tensions between the US and China.**

*by [Will Knight](https://www.technologyreview.com/profile/will-knight/)  February 8, 2019*

**There was a time when the world’s two great superpowers** were obsessed with nuclear weapons technology. Today the flashpoint is between the US and China, and it involves the wireless technology that promises to connect your toaster to the web.     

The two countries are embroiled in a political war over the Chinese telecommunications company Huawei. The Americans have recently stepped up long-standing criticisms, claiming the tech giant has [stolen trade secrets](https://www.bloomberg.com/news/articles/2019-01-28/u-s-planning-to-announce-criminal-charges-related-to-huawei-jrgrda0q) and [committed fraud](https://www.nytimes.com/2019/01/28/us/politics/meng-wanzhou-huawei-iran.html), and that it has [ties to the Chinese government](https://stacks.stanford.edu/file/druid:rm226yb7473/Huawei-ZTE%20Investigative%20Report%20%28FINAL%29.pdf) and its military.

The company denies the charges and has sought to [defend its record](https://www.scmp.com/tech/china-tech/article/2127564/huawei-defends-its-privacy-record-calling-att-snub-big-loss) on privacy and security. Meanwhile, US allies including Great Britain, New Zealand, Australia, Canada, Germany, and Japan have all either imposed restrictions on Huawei’s equipment or are considering doing so, citing national security concerns.

Behind the headlines, though, the spat is also about the coming wave of networking technology known as 5G, and who owns it.

Here are five things you need to know about the technology and its role in the tensions.

### WHAT IS 5G?

Rather than a protocol or device, 5G refers to an array of networking technologies meant to work in concert to connect everything from self-driving cars to home appliances over the air. It’s expected to provide bandwidth of up to 20 gigabits per second—enough to download high-definition movies instantly and use virtual and augmented reality. On your smartphone.

The first 5G smartphones and infrastructure arrive this year, but a full transition will take many more years.

### WHY IS IT BETTER?

5G networks operate on two different frequency ranges. In one mode, they will exploit the same frequencies as existing 4G and Wi-Fi networks, while using a more efficient coding scheme and larger channel sizes to achieve a 25% to 50% speed boost. In a second mode, 5G networks will use much higher, millimeter-wave frequencies that can transmit data at higher speeds, albeit over shorter ranges.

Since millimeter waves drop off over short distances, 5G will require more transmitters. A lot of them, sometimes just a few dozen meters apart. Connected devices will hop seamlessly between these transmitters as well as older hardware.

To increase bandwidth, 5G cells also make use of a technology known as massive MIMO (multiple input, multiple output). This allows hundreds of antennas to work in parallel, which increases speeds and will help lower latency to around a millisecond (from about 30 milliseconds in 4G) while letting more devices connect.

Finally, a technology called full duplex will increase data capacity further still by allowing transmitters and devices to send and receive data on the same frequency. This is done using specialized circuits capable of ensuring that incoming and outgoing signals do not interfere with one another.

### WHAT ARE THE SECURITY RISKS?

One of 5G’s biggest security issues is simply how widely it will be used.

5G stands to replace wired connections and open the door for many more devices to be connected and updated via the internet, including home appliances and industrial machines. Even self-driving cars, industrial robots, and hospital devices that rely on 5G’s ever-present, never-lagging bandwidth will be able to run without a hiccup.

As with any brand-new technology, security vulnerabilities are sure to emerge early on. Researchers in Europe have already [identified weak spots](https://arxiv.org/pdf/1806.10360.pdf) in the way cryptographic keys will be exchanged in 5G networks, for example. With so many more connected devices, the risk for data theft and sabotage—what cybersecurity folks call the attack surface—will be that much higher.

Since 5G is meant to be compatible with existing 4G, 3G, and Wi-Fi networks—in some cases using mesh networking that cuts out central control of a network entirely—[existing security issues](http://wp.internetsociety.org/ndss/wp-content/uploads/sites/25/2018/02/ndss2018_02A-3_Hussain_paper.pdf) will also carry over to the new networks. Britain’s GCHQ [is expected](https://www.theregister.co.uk/2019/02/04/huawei_hcsec_gchq_brouhaha/) to highlight security issues with Huawei’s technology, perhaps involving 4G systems, in coming weeks.

With 5G, a layer of control software will help ensure seamless connectivity, create virtual networks, and offer new network features. A network operator might create a private 5G network for a bank, for instance, and the bank could use features of the network to verify the identities of app users.

This software layer will, however, offer new ways for a malicious network operator to snoop on and manipulate data. It may also open up new vectors for attack, while [hardware bugs](https://meltdownattack.com/) could make it possible for users to hop between virtual networks, eavesdropping or stealing data as they do.

### CAN 5G BE MADE SECURE?

These security worries paint a bleak picture—but there are technical solutions to all of them.

Careful use of cryptography can help secure communications in a way that protects data as it flows across different systems and through virtual networks—even guarding it from the companies that own and run the hardware. Such coding schemes can help guard against jamming, snooping, and hacking.

Two research papers offer a good overview of the risks and potential solutions: *[5G Security: Analysis of Threats and Solutions (pdf)](https://www.google.com/search?q=5G+Security%3A+Analysis+of+Threats+and+Solutions&oq=5G+Security%3A+Analysis+of+Threats+and+Solutions&aqs=chrome..69i57j69i61.191j0j4&sourceid=chrome&ie=UTF-8);**[Security for 5G Mobile Wireless Networks (pdf)](https://ieeexplore.ieee.org/document/8125684)*.

“If you do it correctly, you will actually have a more robust network,” says [Muriel Médard](http://www.rle.mit.edu/ncrc/people/), a professor who leads the [Network Coding and Reliable Communications Group](http://www.rle.mit.edu/ncrc/) at MIT.

### WHY IS HUAWEI’S 5G CAUSING SO MUCH CONCERN?

As the world’s biggest supplier of networking equipment and second largest smartphone maker, Huawei is in a prime position to snatch the lion’s share of a 5G market that, [by some estimates](https://www.marketwatch.com/press-release/5g-services-market-worth-12327-billion-by-2025-2018-09-12), could be worth $123 billion in five years’ time.

Stalling the company’s expansion into Western markets could have the convenient side effect of letting competitors catch up. But there are also legitimate security concerns surrounding 5G—and reasons to think it could be problematic for one company to dominate the space.

The US government appears to have decided that it’s simply too risky for a Chinese company to control too much 5G infrastructure.

The focus on Huawei makes sense given the importance of 5G, the new complexity and security challenges, and the fact that the Chinese company is poised to be such a huge player. And given the way Chinese companies are answerable to the government, Huawei’s apparent connections with the Chinese military and its cyber operations, and the [tightening ties between private industry and the state](https://www.nytimes.com/2018/10/03/business/china-economy-private-enterprise.html), this seems a legitimate consideration.

But the ongoing fight with Huawei also goes to show how vital new technology is to the future of global competition, economic might, and even international security.