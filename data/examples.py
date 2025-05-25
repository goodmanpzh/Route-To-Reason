import json


def get_examples():
    examples = {}
    examples["gsm8k_cot"] = [
        (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
        ),
        (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
        ),
        (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
        ),
        (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
        )
    ]
    examples["gsm8k_pal"] = [
        (
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            '```python\ndef solution():\n    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""\n    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels * bagel_cost\n    money_left = money_initial - money_spent\n    result = money_left\n    return result\n```',
        ),
        (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            '```python\ndef solution():\n    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""\n    golf_balls_initial = 58\n    golf_balls_lost_tuesday = 23\n    golf_balls_lost_wednesday = 2\n    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    result = golf_balls_left\n    return result\n```',
        ),
        (
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            '```python\ndef solution():\n    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""\n    computers_initial = 9\n    computers_per_day = 5\n    num_days = 4  # 4 days between monday and thursday\n    computers_added = computers_per_day * num_days\n    computers_total = computers_initial + computers_added\n    result = computers_total\n    return result\n```',
        ),
    ]
    examples["gsm8k_cod"] = [
        (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "21 - 15 = 6. The answer is 6.",
        ),
        (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "32 + 42 = 74. 74 - 35 = 39. The answer is 39.",
        ),
        (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "3 + 2 = 5. The answer is 5.",
        )
    ]





    # math
    examples["math_cot"] = [
        (
            "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
            "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}}",
        ),
        (
            "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
            "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi",
        ),
        (
            "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
            "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}",
        ),
        (
            "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
            "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0",
        )
    ]


    examples["math_pal"] = [
        (
            "Display the final result in LaTeX.\n\n Find the coefficient of $x^3$ when $3(x^2 - x^3+x) +3(x +2x^3- 3x^2 + 3x^5+x^3) -5(1+x-4x^3 - x^2)$ is simplifie.",
            "```python\nfrom sympy import symbols, simplify\n\ndef solution():\n    x = symbols('x')\n    expr = 3*(x**2 - x**3 + x) + 3*(x + 2*x**3 - 3*x**2 + 3*x**5 + x**3) - 5*(1 + x - 4*x**3 - x**2)\n    simplified_expr = simplify(expr)\n\n    x3_coefficient = simplified_expr.as_coefficients_dict()[x**3]\n    result = x3_coefficient\n    return result\n```",
        ),
        (
            "The surface area of a sphere with radius $r$ is $4\\pi r^2$. Including the area of its circular base, what is the total surface area of a hemisphere with radius 6 cm? Express your answer in terms of $\\pi$.",
            "```python\nimport math\n\ndef solution():\n    radius = 6\n\n    # Surface area of the hemisphere\n    hemisphere_area = 2 * math.pi * radius**2\n\n    # Area of the circular base\n    base_area = math.pi * radius**2\n\n    # Total surface area\n    total_surface_area = hemisphere_area + base_area\n\n    # Formatting the result in LaTeX\n    result = r'{}\\\\pi'.format(total_surface_area / math.pi)\n    return result\n```",
        ),
        (
            "Monica tosses a fair 6-sided die.  If the roll is a prime number, then she wins that amount of dollars (so that, for example, if she rolls 3, then she wins 3 dollars).  If the roll is composite, she wins nothing. Otherwise, she loses 3 dollars. What is the expected value of her winnings on one die toss? Express your answer as a dollar value to the nearest cent.",
            '```python\ndef solution():\n    # Probabilities of each outcome\n    prime_prob = 1 / 6\n    composite_prob = 1 / 3\n    otherwise_prob = 1 / 6\n\n    # Expected value of each outcome\n    prime_expected_value = (2 * prime_prob) + (3 * prime_prob) + (5 * prime_prob)\n    composite_expected_value = 0 * composite_prob\n    otherwise_expected_value = -3 * otherwise_prob\n\n    # Total expected value\n    total_expected_value = prime_expected_value + composite_expected_value + otherwise_expected_value\n\n    # Dollar value to the nearest cent\n    result = "{:.2f}".format(total_expected_value)\n    return result\n```',
        ),
        (
            "Given $\\mathbf{a} = \\begin{pmatrix} -7 \\\\ 0 \\\\ 1 \\end{pmatrix}$ and $\\mathbf{b} = \\begin{pmatrix} 4 \\\\ 2 \\\\ -1 \\end{pmatrix},$ find $\\mathbf{a} - 3 \\mathbf{b}.$",
            "```python\nimport numpy as np\n\ndef solution()\n    a = np.array([-7, 0, 1])\n    b = np.array([4, 2, -1])\n\n    result = a - 3 * b\n\n    result = r'\\begin{{pmatrix}} {} \\\\ {} \\\\ {} \\end{{pmatrix}}'.format(result[0], result[1], result[2])\n    return result\n```",
        ),
    ]


    examples["math_cod"] = [
        (
            "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
            "Step-by-step minimal draft:\n\
            1. Start at position 0\n\
            2. First hop: 1/3\n\
            3. Remaining: 2/3\n\
            4. Second hop: 2/9\n\
            5. Remaining: 4/9\n\
            6. Third hop: 4/27\n\
            7. Remaining: 23/27\n\
            8. Fourth hop: 8/81\n\
            9. Remaining: 73/81\n\
            10. Fifth hop: 16/243\n\
            11. Add all five hops\n\n\
            Total Distance Hopped:\n\
            1/3 + 2/9 + 4/27 + 8/81 + 16/243\n\
            = 81/243 + 54/243 + 36/243 + 24/243 + 16/243\n\
            = 211/243\n\n\
            The Answer is $\\boxed{\\frac{211}{243}}$"
                    ),
        (
            "The surface area of a sphere with radius $r$ is $4\\pi r^2$. Including the area of its circular base, what is the total surface area of a hemisphere with radius 6 cm? Express your answer in terms of $\\pi$.",
            "Step-by-step minimal draft:\n\
            1. Hemisphere: half of sphere\n\
            2. Sphere surface: 4πr²\n\
            3. Hemisphere curved: 2πr²\n\
            4. Add base: πr²\n\
            5. Total: 3πr²\n\
            6. Use r = 6\n\
            7. Compute: 3π × 36\n\
            8. Result: 108π\n\n\
            The Answer is $\\boxed{108\\pi}$"
                    ),
    ]
    



    # mmlu_stem
    examples["mmlu_stem_cot"] = [
        (
            "Simplify and write the result with a rational denominator: $$\\sqrt{\\sqrt[3]{\\sqrt{\frac{1}{729}}}}$$\nAnswer Choices: (A) \\frac{3\\sqrt{3}}{3} (B) \\frac{1}{3} (C) \\sqrt{3} (D) \\frac{\\sqrt{3}}{3}",
            "Factoring $729=3^6$ and combining the roots $\\frac{1}{2}\\frac{1}{3}\\frac{1}{2}=\\frac{1}{12}$, we get that $\\sqrt{\\sqrt[3]{\\sqrt{\frac{1}{729}}}}=\\left(\frac{1}{3^6}\right)^{\frac{1}{12}}=\frac{1}{3^{\frac{1}{2}}}=\frac{3}{\\sqrt{3}}$. The answer is (D).",
        ),
        (
            "In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\nAnswer Choices: (A) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (B) Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane (C) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (D) Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane",
            "Protein synthesis starts at the ribosome, so we can eliminate (A) and (C). The ribosome is often in the endoplasmic reticulum and moves from there to the Golgi apparatus, where it is modified and packaged into a vesicle. The vesicle then floats to the plasma membrane and is secreted. The answer is (D).",
        ),
        (
            "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\nAnswer Choices: (A) 10 W (B) 30 W (C) 60 W (D) 240 W",
            "Rate of energy usage is known as power; in an dissipative electrical circuit, power is given by voltage times current. So in our case, the power is 120 V times 2 amps, or 240 W. The answer is (D).",
        ),
        (
            "Which of the following is considered an acid anhydride?\nAnswer Choices: (A) HCl (B) H2SO3 (C) SO2 (D) Al(NO3)3",
            "An acid anhydride is a compound that is derived by removing water from an acid. The chemical formula for water is H2O, which means that we need to determine which of these options, when combined with H2O, forms an acid. SO2, or Sulfur dioxide, when combined with H2O, makes H2SO4, or sulfuric acid. The answer is (C).",
        ),
        (
            'What is the output of "abc"[::-1] in Python 3? \nAnswer Choices: (A) Error (B) abc (C) cba (D) c',
            'We know that the slicing operator [::-1] takes all of the elements in the string in reverse order, so we reverse the order of the string "abc", resulting in "cba". The answer is (C).',
        ),
    ]

    examples["mmlu_stem_pal"] = [
        (
            "Simplify and write the result with a rational denominator: $$\\sqrt{\\sqrt[3]{\\sqrt{\\frac{1}{729}}}}$$\nAnswer Choices: (A) \\frac{3\\sqrt{3}}{3} (B) \\frac{1}{3} (C) \\sqrt{3} (D) \\frac{\\sqrt{3}}{3}",
            "```python\nfrom sympy import Rational, sqrt, root, simplify\n\ndef solution():\n    # Represent the innermost value as a Rational number\n    base = Rational(1, 729)\n    # First, take the square root of the base\n    first_step = sqrt(base)\n    # Then take the cube root of that result\n    second_step = root(first_step, 3)\n    # Finally, take the square root again\n    expr = sqrt(second_step)\n    # Simplify the expression\n    simplified = simplify(expr)\n    # Match to the correct multiple choice\n    if simplified == Rational(1, 3):\n        return 'Answer: (B)'\n    elif simplified == sqrt(3):\n        return 'Answer: (C)'\n    elif simplified == sqrt(3)/3:\n        return 'Answer: (D)'\n    return 'Answer not found'\n```"
        ),
        (
            "In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\nAnswer Choices: (A) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (B) Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane (C) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (D) Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane",
            "```python\ndef solution():\n    # Correct biological order for secretory protein trafficking\n    correct_path = ['Ribosome', 'rough ER', 'Golgi apparatus', 'secretory vesicle', 'plasma membrane']\n    # Map each answer choice to its sequence\n    options = {\n        'A': ['Plasma membrane', 'Golgi apparatus', 'ribosome', 'secretory vesicle', 'rough ER'],\n        'B': ['Ribosome', 'Golgi apparatus', 'rough ER', 'secretory vesicle', 'plasma membrane'],\n        'C': ['Plasma membrane', 'Golgi apparatus', 'ribosome', 'secretory vesicle', 'rough ER'],\n        'D': ['Ribosome', 'rough ER', 'Golgi apparatus', 'secretory vesicle', 'plasma membrane']\n    }\n    # Compare each option with the correct path\n    for key, value in options.items():\n        if value == correct_path:\n            return f'Answer: ({key})'\n    return 'Answer not found'\n```"
        ),
        (
            "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\nAnswer Choices: (A) 10 W (B) 30 W (C) 60 W (D) 240 W",
            "```python\ndef solution():\n    # Voltage and current given in the problem\n    voltage = 120  # volts\n    current = 2    # amps\n    # Power is voltage * current (P = V * I)\n    power = voltage * current  # watts\n    # Determine the correct answer based on power\n    if power == 10:\n        return 'Answer: (A)'\n    elif power == 30:\n        return 'Answer: (B)'\n    elif power == 60:\n        return 'Answer: (C)'\n    elif power == 240:\n        return 'Answer: (D)'\n    return 'Answer not found'\n```"
        ),
        (
            "Which of the following is considered an acid anhydride?\nAnswer Choices: (A) HCl (B) H2SO3 (C) SO2 (D) Al(NO3)3",
            "```python\ndef solution():\n    # SO2 reacts with H2O to form H2SO3, a typical acid.\n    # This qualifies SO2 as an acid anhydride.\n    acid_anhydride_candidates = {\n        'A': 'HCl',         # Strong acid, but not anhydride\n        'B': 'H2SO3',       # Actual acid, not its anhydride\n        'C': 'SO2',         # Correct: forms H2SO3 when combined with water\n        'D': 'Al(NO3)3'     # A salt, not an acid anhydride\n    }\n    return 'Answer: (C)'\n```"
        ),
        (
            'What is the output of "abc"[::-1] in Python 3? \nAnswer Choices: (A) Error (B) abc (C) cba (D) c',
            "```python\ndef solution():\n    # Python string slicing with step -1 reverses the string\n    result = 'abc'[::-1]\n    # Match the result with answer choices\n    if result == 'abc':\n        return 'Answer: (B)'\n    elif result == 'cba':\n        return 'Answer: (C)'\n    elif result == 'c':\n        return 'Answer: (D)'\n    else:\n        return 'Answer: (A)'\n```"
        ),
    ]



    examples["mmlu_stem_cod"] = [
        (
            "Simplify and write the result with a rational denominator: $$\\sqrt{\\sqrt[3]{\\sqrt{\\frac{1}{729}}}}$$\nAnswer Choices: (A) \\frac{3\\sqrt{3}}{3} (B) \\frac{1}{3} (C) \\sqrt{3} (D) \\frac{\\sqrt{3}}{3}",
            "Step-by-step minimal draft:\n\
            1. Write 729 as power: 729 = 3^6\n\
            2. Expression becomes: sqrt(cbrt(sqrt(1/3^6)))\n\
            3. Combine roots: exponents multiply: (1/3^6)^(1/2 * 1/3 * 1/2) = (1/3^6)^(1/12)\n\
            4. Simplify power: (1/3^6)^(1/12) = 1/3^(6/12) = 1/3^(1/2)\n\
            5. This is 1/√3 = √3/3\n\
            6. Match to answer: (D) √3/3\n\n\
            The Answer is (D)"
        ),
        (
            "In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\nAnswer Choices: (A) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (B) Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane (C) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER (D) Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane",
            "Step-by-step minimal draft:\n\
            1. Protein synthesis starts at ribosome\n\
            2. Secretory proteins go into rough ER\n\
            3. Then sent to Golgi for modification\n\
            4. Packed into secretory vesicle\n\
            5. Vesicle fuses with plasma membrane\n\
            6. Correct sequence: Ribosome → rough ER → Golgi → vesicle → membrane\n\
            7. Match with options\n\n\
            The Answer is (D)"
        ),
        (
            "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\nAnswer Choices: (A) 10 W (B) 30 W (C) 60 W (D) 240 W",
            "Step-by-step minimal draft:\n\
            1. Power = Voltage × Current\n\
            2. Voltage = 120 V\n\
            3. Current = 2 A\n\
            4. Compute power: 120 × 2 = 240\n\
            5. Units are watts\n\
            6. Match with answer choices\n\n\
            The Answer is (D)"
        ),
        (
            "Which of the following is considered an acid anhydride?\nAnswer Choices: (A) HCl (B) H2SO3 (C) SO2 (D) Al(NO3)3",
            "Step-by-step minimal draft:\n\
            1. Acid anhydride = oxide that forms acid with water\n\
            2. Check SO2: SO2 + H2O → H2SO3\n\
            3. SO2 forms acid, qualifies as acid anhydride\n\
            4. HCl is already an acid\n\
            5. H2SO3 is the acid formed, not anhydride\n\
            6. Al(NO3)3 is a salt\n\
            7. Only SO2 matches definition\n\n\
            The Answer is (C)"
        ),
        (
            'What is the output of "abc"[::-1] in Python 3? \nAnswer Choices: (A) Error (B) abc (C) cba (D) c',
            "Step-by-step minimal draft:\n\
            1. 'abc'[::-1] uses slicing\n\
            2. Step -1 reverses the string\n\
            3. Reversed string is 'cba'\n\
            4. Match with options\n\n\
            The Answer is (C)"
        ),
    ]


    # olympiadbench
    examples["olympiadbench_cot"] = [
        (
            "Let $T=11$. Compute the value of $x$ that satisfies $\\sqrt{20+\\sqrt{T+x}}=5$.",
            "We begin by squaring both sides of the equation: $(\\sqrt{20+\\sqrt{T+x}})^2 = 5^2$, so $20+\\sqrt{T+x}=25$. Subtracting 20 from both sides gives $\\sqrt{T+x}=5$. Squaring again yields $T+x=25$, so $x=25-T$. Since $T=11$, we have $x=14$. The answer is 14."
        ),
        (
            "The sum of the interior angles of an $n$-gon equals the sum of the interior angles of a pentagon plus the sum of the interior angles of an octagon. Compute $n$.",
            "The interior angle sum of an $n$-gon is given by $180(n-2)$. For a pentagon, it's $180(5-2)=540^\circ$, and for an octagon, $180(8-2)=1080^\circ$. The sum is $540+1080=1620^\circ$. Setting up the equation: $180(n-2)=1620$, so $n-2=9$, which means $n=11$. The answer is 11."
        ),
    ]


    examples["olympiadbench_pal"] = [
        (
            "Let $T=11$. Compute the value of $x$ that satisfies $\\sqrt{20+\\sqrt{T+x}}=5$.",
            "```python\ndef solution():\n    # Define T and the target equation\n    T = 11\n    # Start by solving the outer square root\n    outer_eq = 5**2  # 25\n    # Now solve the inner square root\n    inner_root = outer_eq - 20  # 5\n    # Now solve for x: sqrt(T + x) = 5 ⇒ T + x = 25\n    x = 25 - T  # 14\n    return x \n```"
        ),
        (
            "The sum of the interior angles of an $n$-gon equals the sum of the interior angles of a pentagon plus the sum of the interior angles of an octagon. Compute $n$.",
            "```python\ndef solution():\n    # Use formula: angle sum = 180 * (n - 2)\n    pentagon_sum = 180 * (5 - 2)  # 540\n    octagon_sum = 180 * (8 - 2)   # 1080\n    total = pentagon_sum + octagon_sum  # 1620\n    # Now solve for n in 180*(n - 2) = 1620\n    n = (1620 / 180) + 2  # 9 + 2\n    return n \n```"
        ),
    ]

    examples["olympiadbench_cod"] = [
        (
            "Let $T=11$. Compute the value of $x$ that satisfies $\\sqrt{20+\\sqrt{T+x}}=5$.",
            "Step-by-step minimal draft:\n\
            1. Start with: √(20 + √(T + x)) = 5\n\
            2. Square both sides: 20 + √(T + x) = 25\n\
            3. Subtract 20: √(T + x) = 5\n\
            4. Square again: T + x = 25\n\
            5. Rearrange: x = 25 - T\n\
            6. Given T = 11, x = 14\n\n\
            The Answer is 14"
        ),
        (
            "The sum of the interior angles of an $n$-gon equals the sum of the interior angles of a pentagon plus the sum of the interior angles of an octagon. Compute $n$.",
            "Step-by-step minimal draft:\n\
            1. Use formula: sum = 180(n - 2)\n\
            2. Pentagon: 180(5 - 2) = 540\n\
            3. Octagon: 180(8 - 2) = 1080\n\
            4. Total: 540 + 1080 = 1620\n\
            5. Set up equation: 180(n - 2) = 1620\n\
            6. Solve: n - 2 = 9 ⇒ n = 11\n\n\
            The Answer is 11"
        ),
    ]

    return examples
