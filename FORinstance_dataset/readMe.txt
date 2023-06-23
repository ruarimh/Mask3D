In the folders you can find:

- Tree data
   These consist in field measured tree properties. DBH will be available for all annotated trees. In addition, in some datasets other variables are available (e.g. tree species, H, V, AGB)
	Variables:
	 plotID: unique identifier for the plots
	 treeID: treeID matching ID numbering in annotated trees 
	 DBH   : diameter (cm) at breast height (1.3 m)
	 treeSP: tree species (see below)
			treeSP
		 1: Picea abies
	 	2: Pinus silvestris
	 	3: Betula pendula
	 	4: Pinus radiata
	 	5: Eucaliptus sp.
		6: Deciduous sp.TUWIEN (not yet defined which species they are)


- Fully annotated point clouds 
   The annotations can be accessed through the following fields:
	treeID
	 plotwise unique identifier annotated trees

	Classification
	 0 : unclassified (scattered points that were not annotated)
	 1: low-vegetation (anything that is not a tree or ground)
	 2: ground
	 3: outpoints (trees outside of the measured/annotated plots)
	 4: stem points
	 5: live-branches (green crown)
	 6: branches

	treeSP
	 1: Picea abies
	 2: Pinus silvestris
	 3: Betula pendula
	 4: Pinus radiata
	 5: Eucaliptus sp.
	 6: Deciduous sp.TUWIEN (not yet defined which species they are)
	
