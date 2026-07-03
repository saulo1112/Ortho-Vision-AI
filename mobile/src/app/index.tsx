import * as Haptics from 'expo-haptics';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { listInferences, predictImage } from '../api/client';
import type { InferenceSummary } from '../api/types';
import { Disclaimer } from '../components/Disclaimer';
import { GlassCard } from '../components/GlassCard';
import { CameraIcon, ClockIcon, GalleryIcon, PulseIcon } from '../components/icons';
import { MaskOverlay } from '../components/MaskOverlay';
import { ScanOverlay } from '../components/ScanOverlay';
import { StatusPill } from '../components/StatusPill';
import { setAnalysis } from '../lib/analysis-store';
import { colors, fonts, radius, spacing } from '../theme/tokens';

interface PickedImage {
  uri: string;
  width: number;
  height: number;
}

export default function HomeScreen() {
  const { width: windowWidth } = useWindowDimensions();
  const [picked, setPicked] = useState<PickedImage | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<InferenceSummary[]>([]);
  const [cameraPermission, requestCameraPermission] = ImagePicker.useCameraPermissions();

  useFocusEffect(
    useCallback(() => {
      listInferences(6)
        .then((res) => setRecent(res.items.filter((item) => item.thumbnail_b64)))
        .catch(() => setRecent([]));
    }, []),
  );

  const analyze = async (image: PickedImage) => {
    setPicked(image);
    setError(null);
    setAnalyzing(true);
    try {
      const response = await predictImage(image.uri);
      setAnalysis({
        imageUri: image.uri,
        imageWidth: image.width,
        imageHeight: image.height,
        response,
      });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      router.push('/result');
    } catch (err) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const toPicked = (asset: ImagePicker.ImagePickerAsset): PickedImage => ({
    uri: asset.uri,
    width: asset.width,
    height: asset.height,
  });

  const pickFromLibrary = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.9,
    });
    if (!result.canceled) analyze(toPicked(result.assets[0]));
  };

  const takePhoto = async () => {
    if (!cameraPermission?.granted) {
      const response = await requestCameraPermission();
      if (!response.granted) return;
    }
    const result = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.9 });
    if (!result.canceled) analyze(toPicked(result.assets[0]));
  };

  const previewWidth = windowWidth - spacing.lg * 2;
  const previewHeight = picked
    ? Math.min(previewWidth * (picked.height / picked.width), previewWidth * 1.3)
    : 0;

  return (
    <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.brand}>OrthoVision AI</Text>
            <Text style={styles.tagline}>Orthopedic implant segmentation</Text>
          </View>
          <Pressable
            onPress={() => router.push('/history')}
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
            hitSlop={8}
          >
            <ClockIcon color={colors.textSecondary} />
          </Pressable>
        </View>

        <StatusPill />

        <GlassCard style={styles.hero}>
          <View style={styles.heroIconWrap}>
            <PulseIcon color={colors.primary} size={26} />
          </View>
          <Text style={styles.heroTitle}>Analyze a radiograph</Text>
          <Text style={styles.heroCopy}>
            Capture or upload an X-ray. The model segments intramedullary nails, screwed plates
            and joint prostheses with per-instance confidence.
          </Text>
          <View style={styles.actions}>
            <Pressable
              onPress={takePhoto}
              disabled={analyzing}
              style={({ pressed }) => [styles.primaryButton, pressed && styles.pressed]}
            >
              <CameraIcon color="#06251F" size={19} strokeWidth={2} />
              <Text style={styles.primaryLabel}>Take photo</Text>
            </Pressable>
            <Pressable
              onPress={pickFromLibrary}
              disabled={analyzing}
              style={({ pressed }) => [styles.secondaryButton, pressed && styles.pressed]}
            >
              <GalleryIcon color={colors.primary} size={19} />
              <Text style={styles.secondaryLabel}>Library</Text>
            </Pressable>
          </View>
        </GlassCard>

        {picked && (
          <View style={styles.previewWrap}>
            <MaskOverlay
              width={previewWidth}
              height={previewHeight}
              detections={[]}
              imageUri={picked.uri}
            />
            {analyzing && (
              <View style={[StyleSheet.absoluteFill, styles.previewOverlayClip]}>
                <ScanOverlay height={previewHeight} />
              </View>
            )}
          </View>
        )}

        {error && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
            {picked && (
              <Pressable onPress={() => analyze(picked)} hitSlop={8}>
                <Text style={styles.retry}>Try again</Text>
              </Pressable>
            )}
          </View>
        )}

        {recent.length > 0 && (
          <View style={styles.recentSection}>
            <Text style={styles.sectionTitle}>Recent analyses</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.recentRow}>
                {recent.map((item) => (
                  <Pressable
                    key={item.inference_id}
                    onPress={() => router.push(`/inference/${item.inference_id}`)}
                    style={({ pressed }) => pressed && styles.pressed}
                  >
                    <Image
                      source={{ uri: `data:image/jpeg;base64,${item.thumbnail_b64}` }}
                      style={styles.recentThumb}
                      contentFit="cover"
                    />
                  </Pressable>
                ))}
              </View>
            </ScrollView>
          </View>
        )}

        <View style={styles.footer}>
          <Disclaimer />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.lg, gap: spacing.md },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  brand: { color: colors.text, fontSize: 28, fontWeight: '700', fontFamily: fonts?.rounded },
  tagline: { color: colors.textSecondary, fontSize: 13, marginTop: 2, fontFamily: fonts?.sans },
  iconButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pressed: { opacity: 0.65 },
  hero: { padding: spacing.lg, gap: spacing.sm },
  heroIconWrap: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: colors.primaryDim,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.xs,
  },
  heroTitle: { color: colors.text, fontSize: 21, fontWeight: '700', fontFamily: fonts?.sans },
  heroCopy: { color: colors.textSecondary, fontSize: 14, lineHeight: 21, fontFamily: fonts?.sans },
  actions: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.sm },
  primaryButton: {
    flex: 1,
    flexDirection: 'row',
    gap: 8,
    height: 50,
    borderRadius: radius.control,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryLabel: { color: '#06251F', fontSize: 15, fontWeight: '700', fontFamily: fonts?.sans },
  secondaryButton: {
    flex: 1,
    flexDirection: 'row',
    gap: 8,
    height: 50,
    borderRadius: radius.control,
    backgroundColor: colors.primaryDim,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryLabel: { color: colors.primary, fontSize: 15, fontWeight: '600', fontFamily: fonts?.sans },
  previewWrap: { borderRadius: 20, overflow: 'hidden' },
  previewOverlayClip: { borderRadius: 20, overflow: 'hidden' },
  errorBox: {
    padding: spacing.md,
    borderRadius: radius.control,
    backgroundColor: 'rgba(248, 113, 113, 0.1)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(248, 113, 113, 0.35)',
    gap: spacing.sm,
  },
  errorText: { color: colors.danger, fontSize: 13, lineHeight: 19, fontFamily: fonts?.sans },
  retry: { color: colors.primary, fontSize: 14, fontWeight: '600', fontFamily: fonts?.sans },
  recentSection: { gap: spacing.sm },
  sectionTitle: { color: colors.text, fontSize: 16, fontWeight: '600', fontFamily: fonts?.sans },
  recentRow: { flexDirection: 'row', gap: spacing.sm },
  recentThumb: {
    width: 74,
    height: 74,
    borderRadius: 14,
    backgroundColor: colors.bgElevated,
  },
  footer: { marginTop: spacing.md },
});
