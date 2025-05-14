import networkx as nx
import numpy as np


class MetricsComputer:
    """
    Compute various metrics related to object detection.

    The user has access to:

        - run_on_batch: compute metrics
        - get_precision
        - get_recall
        - get_f_score

    Attributes
    ----------
    iou_threshold: float
        Threshold to consider for a match between
        a ground truth box and a predicted box
    precision: float
        The precision over the batch
    recall: float
        The recall over the batch
    f_score: float
        The F-score over the batch

    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Init function.

        Parameters
        ----------
        iou_threshold: float
            Threshold to consider for a match between
            a ground truth box and a predicted box

        """
        self.iou_threshold = iou_threshold
        self.precision = 0.0
        self.recall = 0.0
        self.f_score = 0.0

    def run_on_batch(self, predictions: list, ground_truth: list) -> float:
        """
        Run the computation of metrics on a batch of images.

        Parameters
        ----------
        predictions: list
            The list of predictions for this batch
        ground_truth: dict
            The list of ground truth

        Returns
        -------
        float:
            The F-score over the batch

        """
        n_tp = n_fp = n_fn = 0
        for pred, gt in zip(predictions, ground_truth):
            tp, fp, fn = self.run_on_image(pred, gt)
            n_tp += tp
            n_fp += fp
            n_fn += fn
        self.f_score = self._compute_f_score(n_tp, n_fp, n_fn)
        return self.f_score

    def run_on_image(self, predictions: dict, ground_truth: dict):
        """
        Run the computation of metrics on one image.

        Parameters
        ----------
        predictions: list
            The predictions for this batch (dict of boxes, labels, scores)
        ground_truth: list
            The ground truth for this batch (dict of boxes, labels, scores)

        Returns
        -------
        tp: int
            The number of true positives in the image
        fp: int
            The number of false positives in the image
        fn: int
            The number of false negatives in the image

        """
        matches = self._match_boxes(
            predictions["boxes"], ground_truth["boxes"]
        )
        tp = len(matches)
        fp = len(predictions["boxes"]) - tp
        fn = len(ground_truth["boxes"]) - tp
        return tp, fp, fn

    def _match_boxes(self, pred_boxes: list, gt_boxes: list) -> list:
        """
        Match a prediction box to a ground truth box.

        It uses the Maximum Weighted Bipartite Matching algorithm.

        Parameters
        ----------
        pred_boxes: list
            list of predictions bounding boxes
        gt_boxes: list
            list of ground truth bounding boxes

        Returns
        -------
        list:
            The matches

        """
        pred_boxes = pred_boxes.detach().cpu().numpy()
        gt_boxes = np.array(gt_boxes.cpu())

        # Create a bipartite graph
        graph = nx.Graph()
        graph.add_nodes_from(range(len(pred_boxes)), bipartite=0)
        graph.add_nodes_from(range(len(gt_boxes)), bipartite=1)

        # Add edges with weights (IoU values)
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = self._compute_iou(pred_box, gt_box)
                if iou >= self.iou_threshold:
                    graph.add_edge(i, len(pred_boxes) + j, weight=iou)

        # Find the maximum weighted bipartite matching
        matching = nx.max_weight_matching(graph, maxcardinality=True)

        # Convert the matching to a list of pairs
        matches = []
        for u, v in matching:
            if u < len(pred_boxes) and v >= len(pred_boxes):
                matches.append((pred_boxes[u], gt_boxes[v - len(pred_boxes)]))

        return matches

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute the intersection over union of 2 bounding boxes.

        Parameters
        ----------
        bbox1: numpy.ndarray
          First bounding box
        bbox2: numpy.ndarray
          Second bounding box

        Returns
        -------
        float:
          The intersection over union

        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        inter_area = max(0, xb - xa) * max(0, yb - ya)

        # compute the area of both the prediction and ground-truth rectangles
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = float(box1_area + box2_area - inter_area)
        iou = inter_area / union if union != 0 else 0
        return iou

    def _compute_f_score(
        self,
        true_positives: int,
        false_positives: int,
        false_negatives: int
    ) -> float:  # fmt: off
        """
        Compute the F-score, precision, and recall.

        Note: the attributes precision and recall are set in this function

        Parameters
        ----------
        true_positives: int
            Number of true positives
        false_positives: int
            Number of false positives
        false_negatives: int
            Number of false negatives

        Returns
        -------
        float:
            the F-score

        """
        # Compute precision
        if true_positives + false_positives > 0:
            precision = true_positives / float(
                true_positives + false_positives
            )
        else:
            precision = 0.0

        # Compute recall
        if true_positives + false_negatives > 0:
            recall = true_positives / float(true_positives + false_negatives)
        else:
            recall = 0.0

        # Compute F-score
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = 0.0

        self.precision = precision
        self.recall = recall

        f_score = np.float32(f_score)
        return f_score
