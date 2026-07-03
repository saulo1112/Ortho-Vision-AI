import { Image } from 'expo-image';
import { StyleSheet, View } from 'react-native';
import Svg, { Polygon } from 'react-native-svg';

import type { Detection } from '../api/types';
import { classStyle } from '../theme/classes';

interface Props {
  /** Rendered size — the container must match the image's aspect ratio. */
  width: number;
  height: number;
  detections: Detection[];
  imageUri?: string;
  borderRadius?: number;
}

/**
 * Radiograph with the segmentation polygons drawn on top. Polygon points
 * arrive normalized to [0, 1] of the original image, so scaling to the
 * rendered box is a single multiply per axis.
 */
export function MaskOverlay({ width, height, detections, imageUri, borderRadius = 20 }: Props) {
  return (
    <View style={{ width, height, borderRadius, overflow: 'hidden' }}>
      {imageUri ? (
        <Image source={{ uri: imageUri }} style={StyleSheet.absoluteFill} contentFit="fill" />
      ) : null}
      <Svg width={width} height={height} style={StyleSheet.absoluteFill}>
        {detections.map((det) => {
          const { color } = classStyle(det.class_name);
          return det.polygons.map((ring, ringIndex) => {
            const points = ring.map(([x, y]) => `${x * width},${y * height}`).join(' ');
            return (
              <Polygon
                key={`${det.id}-${ringIndex}`}
                points={points}
                fill={`${color}55`}
                stroke={color}
                strokeWidth={2}
                strokeLinejoin="round"
              />
            );
          });
        })}
      </Svg>
    </View>
  );
}
