package seakers.eossrl.actions.assignment;

import seakers.vassarexecheur.search.operators.assigning.RepairInstrumentCountAssigning;
import seakers.vassarheur.ResourcePool;
import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.problems.Assigning.ArchitectureEvaluator;


public class RepairInstrumentCount extends RepairInstrumentCountAssigning {

    private int numInstruments; // number of instruments to remove per satellite
    private int numSatellites; // number of satellites to remove instruments from
    private final double threshold; // Maximum total instrument count to satisfy heuristic
    private AssigningProblem problem; // problem class for architecture evaluation
    private final ResourcePool resourcePool; // used for architecture evaluation
    private final BaseParams params; // used for architecture evaluation
    private final ArchitectureEvaluator evaluator; // used for architecture evaluation

    public RepairInstrumentCount(int numInstruments, int numSatellites, double threshold, AssigningProblem problem, ResourcePool resourcePool, BaseParams params, ArchitectureEvaluator evaluator) {
        super(numInstruments, numSatellites, threshold, problem, resourcePool, evaluator, params);
        this.numInstruments = numInstruments;
        this.numSatellites = numSatellites;
        this.threshold = threshold;
        this.problem = problem;
        this.resourcePool = resourcePool;
        this.params = params;
        this.evaluator = evaluator;
    }
    
    // All parent methods can be used, below are modified versions of the methods to override them if needed

    /*
     * Methods: 
     *  1. Solution[] evolve(Solution[] sols)
     *  2. int getArity()
     */
}