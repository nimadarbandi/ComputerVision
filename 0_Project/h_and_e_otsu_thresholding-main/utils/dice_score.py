"""Defines the Sørensen–Dice coefficient"""

def dice_score(mask_1, mask_2):

    return 2 * (mask_1 * mask_2).sum()/(mask_1.sum() + mask_2.sum())