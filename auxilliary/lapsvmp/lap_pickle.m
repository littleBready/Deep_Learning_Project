function L = lap_pickle(options,adjlist)  

W = adj_pickle(adjlist);
D = sum(W,2);

if options.LaplacianNormalize == 0
    L = spdiags(D,0,speye(size(W,1)))-W; % L = D-W
else 
    D(D~=0)=sqrt(1./D(D~=0));
    D=spdiags(D,0,speye(size(W,1)));
    W=D*W*D;
    L=speye(size(W,1))-W; % L = I-D^-1/2*W*D^-1/2
end

if options.LaplacianDegree>1
    L=mpower(L,options.LaplacianDegree);    
end
