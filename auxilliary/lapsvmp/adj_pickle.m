function A = adj_pickle (adjlist)

[m,n] = size(adjlist);

A = zeros (m);

for i = 1:m
    for j = 1:n
        k = adjlist(i,j) + 1;
        if ~isnan(k)
            A(i,k) = 1;
            A(k,i) = 1;
        end
    end
end
