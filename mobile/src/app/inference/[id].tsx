import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { getInference } from '../../api/client';
import type { InferenceDetail } from '../../api/types';
import { DetectionCard } from '../../components/DetectionCard';
import { ChevronLeftIcon } from '../../components/icons';
import { MaskOverlay } from '../../components/MaskOverlay';
import { timeAgo } from '../../lib/format';
import { colors, fonts, radius, spacing } from '../../theme/tokens';

export default function InferenceDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { width: windowWidth } = useWindowDimensions();
  const [detail, setDetail] = useState<InferenceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getInference(id)
      .then(setDetail)
      .catch((err) => setError(err instanceof Error ? err.message : 'Could not load analysis'));
  }, [id]);

  const renderWidth = windowWidth - spacing.lg * 2;
  const aspect = detail ? detail.image.height / detail.image.width : 1;

  return (
    <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.headerRow}>
          <Pressable
            onPress={() => router.back()}
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
            hitSlop={8}
          >
            <ChevronLeftIcon color={colors.textSecondary} />
          </Pressable>
          <Text style={styles.title}>Analysis</Text>
          <View style={styles.headerSpacer} />
        </View>

        {!detail && !error && <ActivityIndicator color={colors.primary} style={styles.loader} />}
        {error && <Text style={styles.error}>{error}</Text>}

        {detail && (
          <>
            <MaskOverlay
              width={renderWidth}
              height={renderWidth * aspect}
              detections={detail.detections}
              imageUri={
                detail.thumbnail_b64 ? `data:image/jpeg;base64,${detail.thumbnail_b64}` : undefined
              }
            />

            <View style={styles.list}>
              {detail.detections.map((det) => (
                <DetectionCard key={det.id} detection={det} />
              ))}
              {detail.detections.length === 0 && (
                <Text style={styles.emptyCopy}>No implants were detected in this image.</Text>
              )}
            </View>

            <View style={styles.metaCard}>
              <MetaRow label="Analyzed" value={timeAgo(detail.created_at)} />
              <MetaRow label="Model" value={detail.model_version} />
              <MetaRow label="Confidence threshold" value={`${detail.conf_threshold}`} />
              <MetaRow
                label="Image"
                value={`${detail.image.width}×${detail.image.height}px`}
              />
              <MetaRow label="Server time" value={`${detail.timing_ms.total} ms`} />
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function MetaRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <View style={styles.metaRow}>
      <Text style={styles.metaLabel}>{label}</Text>
      <Text style={[styles.metaValue, mono && styles.mono]} numberOfLines={1}>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.lg, gap: spacing.md },
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  title: { color: colors.text, fontSize: 18, fontWeight: '700', fontFamily: fonts?.sans },
  iconButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerSpacer: { width: 42, height: 42 },
  pressed: { opacity: 0.65 },
  loader: { marginTop: 60 },
  error: { color: colors.danger, textAlign: 'center', marginTop: 40, fontFamily: fonts?.sans },
  list: { gap: spacing.sm },
  emptyCopy: { color: colors.textSecondary, fontSize: 13, textAlign: 'center', fontFamily: fonts?.sans },
  metaCard: {
    borderRadius: radius.control,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
    paddingHorizontal: spacing.md,
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    gap: spacing.md,
  },
  metaLabel: { color: colors.textTertiary, fontSize: 13, fontFamily: fonts?.sans },
  metaValue: { color: colors.textSecondary, fontSize: 13, flexShrink: 1, fontFamily: fonts?.sans },
  mono: { fontFamily: fonts?.mono },
});
