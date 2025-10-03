package com.digitrecognition.models;

/**
 * Helper class to store neighbor information for k-NN algorithms.
 * Contains the label of a neighbor and its distance from the query point.
 * Implements Comparable to enable sorting by distance.
 * 
 * @author Dumitru Nirca
 */
public class NeighborDistance implements Comparable<NeighborDistance> {
    private final int label;
    private final double distance;
    
    /**
     * Creates a new NeighborDistance with the given label and distance.
     * 
     * @param label The class label of the neighbor
     * @param distance The distance from the query point
     */
    public NeighborDistance(int label, double distance) {
        this.label = label;
        this.distance = distance;
    }
    
    /**
     * Gets the label of this neighbor.
     * 
     * @return The class label
     */
    public int getLabel() {
        return label;
    }
    
    /**
     * Gets the distance of this neighbor from the query point.
     * 
     * @return The distance value
     */
    public double getDistance() {
        return distance;
    }
    
    /**
     * Compares this neighbor with another based on distance.
     * Used for sorting neighbors by distance in ascending order.
     * 
     * @param other The other NeighborDistance to compare with
     * @return Negative if this distance is smaller, positive if larger, 0 if equal
     */
    @Override
    public int compareTo(NeighborDistance other) {
        return Double.compare(this.distance, other.distance);
    }
    
    @Override
    public String toString() {
        return String.format("NeighborDistance{label=%d, distance=%.3f}", label, distance);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        NeighborDistance that = (NeighborDistance) obj;
        return label == that.label && 
               Double.compare(that.distance, distance) == 0;
    }
    
    @Override
    public int hashCode() {
        int result = label;
        long temp = Double.doubleToLongBits(distance);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
}
