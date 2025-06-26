# rules that are being made while making of dataset

# Change how @birule is showed in the repl
trigs_rules = [@birule tan(~x) ~ sin(~x) / cos(~x)
    @birule sec(~x) ~ one(~x) / cos(~x)
    @birule csc(~x) ~ one(~x) / sin(~x)
    @birule cot(~x) ~ cos(~x) / sin(~x)

    @birule sin(~x)^2 ~ 1 + -1*cos(~x)^2
    @birule sec(~x)^2 ~ 1 + tan(~x)^2
    @birule csc(~x)^2 ~ 1 + cot(~x)^2 
    @birule sin(2*~x) ~ 2*sin(~x)*cos(~x)  
    @birule cos(2*~x) ~ 2*^(cos(~x), 2) + -1
    @birule tan(2*~x) ~ (2*tan(~x)) / (1 + -1*tan(~x)^2)
    @birule sin(2*~x) ~ (2*tan(~x)) / (1 + tan(~x)^2)
    @birule cos(2*~x) ~ (1 + -1 * tan(~x)^2) / (1 + tan(~x)^2)

    @birule sin(3*~x) ~ 3sin(~x) + -1*4sin(~x)^3
    @birule cos(3*~x) ~ 4cos(~x)^3 + -1*3cos(~x)
    @birule tan(3*~x) ~ (3tan(~x) + -1*tan(~x)^3) / (1 + -1*3tan(~x)^2)

    # product identities
    @birule sin(~x) * cos(~y) ~ (sin(~x + ~y) + sin(~x + -1*~y)) / 2
    @birule cos(~x) * cos(~y) ~ (cos(~x + ~y) + cos(~x + -1*~y)) / 2
    @birule sin(~x) * sin(~y) ~ -(cos(~x + ~y) + -1*cos(~x + -1*~y)) / 2

    # sum to product Identities
    @birule sin(~x) + sin(~y) ~ 2 * sin((~x + ~y)/2) * cos((~x + -1*~y)/2)
    @birule sin(~x) + -1 * sin(~y) ~ 2 * cos((~x + ~y)/2) * sin((~x + -1*~y)/2)
    @birule cos(~x) + cos(~y) ~ 2 * cos((~x + ~y)/2) * cos((~x + -1*~y)/2)
    @birule cos(~x) + -1 * cos(~y) ~ -2 * sin((~x + ~y)/2) * sin((~x + -1*~y)/2)
]


# trigs_rules[5](sin(x)^2)
# trigs_rules[5](1 - (cos(x)^2))

# expr = 2 * cos(x)^2 - 1
# yup(expr)
# dump(expr)
# trigs_rules[9](cos(2x))

# trigs_rules[end](cos(2x))

# # Check why the inverse of this is not working
# yup = @rule (1 + -1 * tan(~x)^2) / (1 + tan(~x)^2) => cos(2*~x)
# yup((1 - (tan(x)^2)) / (1 + tan(x)^2))