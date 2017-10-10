function A = adj_gtruth (gtruth)

n = length(gtruth);

A = zeros (n);

for i = 2:n
    for j = 1:(i-1)
        if gtruth(i) == gtruth(j)
            A(i,j) = 1;
            A(j,i) = 1;
        end
    end
end
